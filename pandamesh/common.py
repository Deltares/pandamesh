from __future__ import annotations

import functools
import operator
import warnings
from typing import Any, Sequence, Set, Tuple, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api.types import is_integer_dtype


class MaybeGmsh:  # pragma: no cover
    """Gmsh is an optional dependency."""

    def __init__(self):
        try:
            import gmsh

            self.gmsh = gmsh
            self.ok = True
            self.error = None
        except ImportError:
            self.gmsh = None
            self.ok = False
            self.error = ImportError("Gmsh is required for this functionality")
        except ValueError:
            self.gmsh = None
            self.ok = False
            self.error = RuntimeError(
                "Gmsh must run in the main thread of the interpreter"
            )

    def __getattr__(self, name: str):
        if self.ok:
            return getattr(self.gmsh, name)
        else:
            raise self.error


gmsh = MaybeGmsh()
BoolArray = np.ndarray
IntArray = np.ndarray
FloatArray = np.ndarray
GeometryArray = np.ndarray
coord_dtype = np.dtype([("x", np.float64), ("y", np.float64)])
Geometries = TypeVar("Geometries", GeometryArray, gpd.GeoSeries)

POINT = shapely.GeometryType.POINT
LINESTRING = shapely.GeometryType.LINESTRING
LINEARRING = shapely.GeometryType.LINEARRING
POLYGON = shapely.GeometryType.POLYGON
GEOM_NAMES = {v: k for k, v in shapely.GeometryType.__members__.items()}


def repr(obj: Any) -> str:
    strings = [type(obj).__name__]
    for k, v in obj.__dict__.items():
        if k.startswith("_"):
            k = k[1:]
        if isinstance(v, np.ndarray):
            s = f"    {k} = np.ndarray with shape {v.shape}"
        else:
            s = f"    {k} = {v}"
        strings.append(s)
    return "\n".join(strings)


def flatten(seq: Sequence[Any]):
    return functools.reduce(operator.concat, seq)


def flatten_geometry(geom):
    """Recursively flatten geometry collections."""
    if hasattr(geom, "geoms"):
        return [g for subgeom in geom.geoms for g in flatten_geometry(subgeom)]
    else:
        return [geom]


def flatten_geometries(geometries: Sequence) -> GeometryArray:
    """Flatten geometry collections."""
    flattened = []
    for geom in geometries:
        flattened.extend(flatten_geometry(geom))
    return np.array(flattened)


def check_geodataframe(
    features: gpd.GeoDataFrame, required_columns: Set[str], check_index: bool = False
) -> None:
    if not isinstance(features, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected GeoDataFrame, received instead: {type(features).__name__}"
        )

    if len(features) == 0:
        raise ValueError("Dataframe is empty")

    missing = required_columns - set(features.columns)
    if missing:
        raise ValueError(
            f"These column(s) are required but are missing: {', '.join(missing)}"
        )
    if check_index:
        if not is_integer_dtype(features.index):
            raise ValueError(
                f"geodataframe index is not integer typed, received: {features.index.dtype}"
            )
        if features.index.duplicated().any():
            raise ValueError(
                "geodataframe index contains duplicates, call .reset_index()"
            )


def intersecting_features(features, feature_type) -> Tuple[IntArray, IntArray]:
    tree = shapely.STRtree(geoms=features)
    if feature_type == "polygon":  # TODO: might not be necessary
        target = features.buffer(-1.0e-6)
    else:
        target = features
    i, j = tree.query(geometry=target, predicate="intersects")
    # Intersection matrix is symmetric, and contains i==j (diagonal)
    keep = j > i
    return i[keep], j[keep]


def check_intersection(features: gpd.GeoSeries, feature_type: str) -> None:
    if len(features) <= 1:
        return
    index_a, index_b = intersecting_features(features, feature_type)
    n_overlap = len(index_a)
    if n_overlap > 0:
        message = "\n".join([f"{a} with {b}" for a, b in zip(index_a, index_b)])
        raise ValueError(
            f"{n_overlap} cases of intersecting {feature_type} detected:\n{message}"
        )


def check_features(features: gpd.GeoSeries, feature_type) -> None:
    """Check whether features are simple."""
    is_complex = ~features.is_simple
    if is_complex.any():
        n_complex = is_complex.sum()
        index = features.index[is_complex].to_numpy()
        raise ValueError(
            f"{n_complex} cases of complex {feature_type} detected.\nThese "
            f"features contain self intersections:\n{index}"
        )
    return


def compute_intersections(segment_linestrings, i: IntArray, j: IntArray):
    check = j > i
    i = i[check]
    j = j[check]

    intersections = shapely.intersection(
        segment_linestrings[i],
        segment_linestrings[j],
    )
    coordinates, index = shapely.get_coordinates(intersections, return_index=True)
    coordinates, unique_index = np.unique(coordinates, return_index=True, axis=0)
    index = index[unique_index]
    return i[index], j[index], coordinates


def linework_intersections(
    polygons: gpd.GeoSeries,
    lines: gpd.GeoSeries,
    tolerance: float,
) -> FloatArray:
    """Compute intersections between polygon boundary and line segments."""
    outer_rings = polygons.exterior
    inner_rings = np.array(flatten(polygons.interiors))
    linework = np.concatenate((lines, outer_rings, inner_rings))

    coordinates, line_index = shapely.get_coordinates(linework, return_index=True)
    segments = np.stack((coordinates[:-1, :], coordinates[1:]), axis=1)
    keep = np.diff(line_index) == 0
    segments = segments[keep]
    segment_to_line = line_index[1:][keep]
    segment_linestrings = shapely.linestrings(
        segments.reshape((-1, 2)), indices=np.repeat(np.arange(len(segments)), 2)
    )

    # We only use the STRtree to check bounding boxes. It does return the
    # intersection result, even if we specify the predicate.
    rtree = shapely.STRtree(segment_linestrings)
    i, j = rtree.query(segment_linestrings)

    # Compute intersections
    # Skip self-intersections
    keep = segment_to_line[i] != segment_to_line[j]
    i = i[keep]
    j = j[keep]
    i, j, intersections = compute_intersections(segment_linestrings, i, j)
    segments_i = segments[i]
    segments_j = segments[j]

    # Find minimum squared distance to vertices of intersecting edges. If the
    # minimum distance is (approximately) zero, the intersection is represented
    # by a vertex already.
    distance_i = np.linalg.norm(
        intersections[:, np.newaxis, :] - segments_i, axis=2
    ).min(axis=1)
    distance_j = np.linalg.norm(
        intersections[:, np.newaxis, :] - segments_j, axis=2
    ).min(axis=1)
    unresolved = (distance_i > tolerance) | (distance_j > tolerance)
    return intersections[unresolved]


def _triangles(rings: GeometryArray) -> FloatArray:
    coords = shapely.get_coordinates(rings)
    n = shapely.get_num_coordinates(rings)

    polygon_ids = np.repeat(np.arange(len(rings)), n)

    # Calculate the local index within each polygon
    local_indices = np.arange(n.sum()) % n[polygon_ids]

    # Get rid of the last vertex, since this is the same as the first in a
    # linearring.
    lengths = n - 1
    discard_last = local_indices != lengths[polygon_ids]
    local_indices = local_indices[discard_last]
    polygon_ids = polygon_ids[discard_last]

    # Create shifts for the three consecutive vertices
    shifts = np.column_stack(
        (
            local_indices,
            (local_indices + 1) % lengths[polygon_ids],
            (local_indices + 2) % lengths[polygon_ids],
        )
    )
    indices = (np.cumsum(n) - n)[polygon_ids, np.newaxis] + shifts
    return coords[indices]


def _compute_cross_product(triangles: FloatArray) -> FloatArray:
    ab = triangles[:, 1] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    return ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]


def _find_proximate_points(rings: GeometryArray, minimum_spacing: float) -> FloatArray:
    # This function identifies points at very short distances of each other, or
    # two edges that nearly intersect each other: i.e. a sliver triangle. In
    # case of a sliver, the distance of the last vertex (c) to the first edge
    # (a-b) is very small, i.e. the height of the triangle is very small with
    # respect to base (a-b).
    triangles = _triangles(rings)
    # Check convex / concave. Note shapely canonical form: exterior is
    # clockwise (CW), interior is counter-clockwise (CCW).
    # We need to worry about CONVEX angles in the CLOCKWISE exterior;
    # we need to worry about CONCAVE angles in the CCW interior.
    # Both therefore need positive cross product values.
    cross_product = _compute_cross_product(triangles)

    # Identify non-monotone triangles: for a monotone triangle, the angle at
    # the second vertex will always be obtuse (90 < angle < 180), and the
    # distance a-c is always larger than a-b. Hence, to find a shortest
    # segment, we need only compare a-b and b-c; monotonic collinear points are
    # allowed, but have a height of 0.
    # For non-monotone triangles or non-monotone collinear points, we need to
    # consider the height of the triangle, since in that case, point c might be
    # in between a and b (e.g. along the x-axis).
    d_ab = triangles[:, 1, :] - triangles[:, 0, :]
    d_bc = triangles[:, 2, :] - triangles[:, 1, :]
    d_ca = triangles[:, 2, :] - triangles[:, 0, :]
    non_monotone = ((np.sign(d_ab) * np.sign(d_bc)) < 0).any(axis=1)
    check_height = (cross_product <= 0) & non_monotone
    # Euclidian distance.
    a = np.linalg.norm(d_ab, axis=-1)
    b = np.linalg.norm(d_bc, axis=-1)
    c = np.linalg.norm(d_ca, axis=-1)
    # Semiperimeter and Heron's formula.
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    height = (2 * area) / a

    # Use only distance a and b if exterior angle is convex and points are
    # monotonic.
    distance = np.minimum(a, b)
    # Use triangle height otherwise.
    distance[check_height] = height[check_height]
    return triangles[distance <= minimum_spacing, 1, :]


def find_proximate_perimeter_points(
    geometry: gpd.GeoSeries, minimum_spacing: float = 1.0e-3
) -> gpd.GeoSeries:
    """
    Detect vertices in polygon perimeters that are very close to each other.

    Note that dangling edges can be detected through self-intersection: whether
    a geometry is simple or not. However, some slivers will almost form a
    dangling edge, where the sliver still have a very small thickness. This may
    result in problems during mesh generation, as tiny triangles will be
    required locally.

    Note that sliver concavities are allowed: the vertex spacing **along** the
    perimeter is not necessarily small.

    Parameters
    ----------
    geometry : geopandas.Geoseries
        Points, lines, polygons.
    minimum_spacing : float, default is 1.0e-3.
        The minimum allowed distance between vertices, or the minimum width of
        slivers.
    """

    # Normalize forces strict canonical form:
    # "the coordinates of exterior rings follow a clockwise orientation and
    # interior rings have a counter-clockwise orientation"
    polygons, _, _ = separate_geometry(geometry)
    polygons = shapely.normalize(polygons)
    points = _find_proximate_points(
        rings=polygons.exterior.to_numpy(),
        minimum_spacing=minimum_spacing,
    )
    interiors = flatten(polygons.interiors)
    if len(interiors) > 0:
        interior_points = _find_proximate_points(
            rings=np.atleast_1d(interiors),
            minimum_spacing=minimum_spacing,
        )
        points = np.concatenate((points, interior_points))
    return gpd.GeoSeries(gpd.points_from_xy(*points.T))


def find_edge_intersections(geometry: gpd.Geoseries) -> gpd.GeoSeries:
    """
    Find all unresolved intersections between polygon boundaries, linestring,
    and linearring edges.

    A "resolved" intersection is one where the intersection of two lines is
    represented by a vertex in both lines. Unresolved means: an intersection
    which is not represented by an explicit vertex in the geometries.

    Parameters
    ----------
    geometry: gpd.GeoSeries
        Points, lines, polygons.

    Returns
    -------
    intersections: gpd.GeoSeries
        Locations (points) of intersections.
    """
    if not isinstance(geometry, gpd.GeoSeries):
        raise TypeError(
            f"Expected geopandas.GeoSeries, received: {type(geometry).__name__}"
        )
    polygons, lines, _ = separate_geometry(geometry)
    intersections = linework_intersections(
        polygons,
        lines,
        tolerance=1.0e-6,
    )
    return gpd.GeoSeries(gpd.points_from_xy(*intersections.T))


def check_points(
    points: gpd.GeoSeries,
    polygons: gpd.GeoSeries,
) -> None:
    """Check whether points are contained by a polygon."""
    within = gpd.GeoDataFrame(geometry=points).sjoin(
        df=gpd.GeoDataFrame(geometry=polygons),
        predicate="within",
    )
    n_outside = len(points) - len(within)
    if n_outside != 0:
        raise ValueError(f"{n_outside} points detected outside of a polygon")
    return


def check_linework(
    polygons: gpd.GeoSeries,
    lines: gpd.GeoSeries,
    on_unresolved_intersection: str,
) -> None:
    intersections = linework_intersections(
        polygons.geometry,
        lines.geometry,
        tolerance=1.0e-6,
    )
    n_intersection = len(intersections)
    if n_intersection > 0:
        msg = (
            f"{n_intersection} unresolved intersections between polygon "
            "boundary or line segments.\nRun "
            "pandamesh.find_edge_intersections(gdf.geometry) to identify the "
            "intersection locations.\nIntersections can be resolved using "
            "the pandamesh.Preprocessor."
        )
        if on_unresolved_intersection == "error":
            raise ValueError(msg)
        elif on_unresolved_intersection == "warn":
            warnings.warn(msg)
    return


def check_perimeter_proximate_points(
    polygons: gpd.GeoSeries, minimum_spacing: float
) -> None:
    proximate_points = find_proximate_perimeter_points(polygons, minimum_spacing)
    n_proximate = len(proximate_points)
    if n_proximate > 0:
        raise ValueError(
            f"{n_proximate} proximate points found on polygon perimeters.\n"
            "Run pandamesh.find_perimeter_proximate_points to identify these "
            "points."
        )
    return


def _separate(
    geometry: Geometries,
) -> Tuple[BoolArray, BoolArray, BoolArray]:
    geometry_id = shapely.get_type_id(geometry)
    allowed_types = (POINT, LINESTRING, LINEARRING, POLYGON)
    if not np.isin(geometry_id, allowed_types).all():
        received = ", ".join(
            [GEOM_NAMES[geom_id] for geom_id in np.unique(geometry_id)]
        )
        raise TypeError(
            "GeoDataFrame contains unsupported geometry types. Geometry "
            "should be one of Point, LineString, LinearRing, and Polygon "
            f"geometries. Received: {received}\n"
            "Call geopandas.GeoDataFrame.explode() to explode multi-part "
            "geometries into multiple single geometries."
        )
    return (
        geometry_id == POLYGON,
        (geometry_id == LINESTRING) | (geometry_id == LINEARRING),
        geometry_id == POINT,
    )


def separate_geometry(
    geometry: Geometries,
) -> Tuple[Geometries, Geometries, Geometries]:
    is_polygon, is_line, is_point = _separate(geometry)
    return (
        geometry[is_polygon],
        geometry[is_line],
        geometry[is_point],
    )


def check_polygons(polygons: gpd.GeoSeries, minimum_spacing: float) -> None:
    check_features(polygons, "polygon")
    check_perimeter_proximate_points(polygons, minimum_spacing)
    check_intersection(polygons, "polygon")
    return


def check_lines(lines: gpd.GeoSeries) -> None:
    check_features(lines, "lines")
    return


def separate_geodataframe(
    gdf: gpd.GeoDataFrame,
    intersecting_edges: str,
    minimum_spacing: float,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    acceptable = ("error", "warn", "ignore")
    if intersecting_edges not in acceptable:
        raise ValueError(
            f"intersecting_edges should be one of {acceptable}, "
            f"received: {intersecting_edges}"
        )

    is_polygon, is_line, is_point = _separate(gdf["geometry"])
    polygons = gdf.loc[is_polygon].copy()
    lines = gdf.loc[is_line].copy()
    points = gdf.loc[is_point].copy()

    if len(polygons) == 0:
        raise ValueError("No polygons provided")

    for df in (polygons, lines, points):
        df["cellsize"] = df["cellsize"].astype(float)
        df.crs = None

    check_polygons(polygons.geometry, minimum_spacing)
    check_lines(lines.geometry)
    if intersecting_edges != "ignore":
        check_linework(polygons.geometry, lines.geometry, intersecting_edges)
    check_points(points.geometry, polygons.geometry)

    return polygons, lines, points


def move_origin(
    gdf: gpd.GeoDataFrame,
    xoff: float,
    yoff: float,
):
    if xoff == 0.0 and yoff == 0.0:
        return gdf.copy()
    else:
        moved = gdf.copy()
        moved["geometry"] = gdf["geometry"].translate(xoff=-xoff, yoff=-yoff)
    return moved


def central_origin(
    gdf: gpd.GeoDataFrame, shift_origin: bool
) -> Tuple[gpd.GeoDataFrame, float, float]:
    if shift_origin:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        xoff = 0.5 * (xmin + xmax)
        yoff = 0.5 * (ymin + ymax)
    else:
        xoff = 0.0
        yoff = 0.0
    gdf = move_origin(gdf, xoff, yoff)
    return gdf, xoff, yoff


def to_ugrid(vertices: FloatArray, faces: IntArray) -> "xugrid.Ugrid2d":  # type: ignore # noqa pragma: no cover
    try:
        import xugrid
    except ImportError:
        raise ImportError(
            "xugrid must be installed to return generated result a xugrid.Ugrid2d"
        )
    return xugrid.Ugrid2d(*vertices.T, -1, faces)


def to_geodataframe(vertices: FloatArray, faces: IntArray) -> gpd.GeoDataFrame:
    n_face, n_vertex = faces.shape
    if n_vertex == 3:  # no fill values
        coordinates = vertices[faces]
        geometry = shapely.polygons(coordinates)
    else:  # Possible fill values (-1)
        # Group them by number of vertices.
        valid = faces >= 0
        n_valid = valid.sum(axis=1)
        grouped_by_n_vertex = pd.DataFrame(
            {"i": np.arange(n_face), "n": n_valid}
        ).groupby("n")
        geometries = []
        for n_vertex, group in grouped_by_n_vertex:
            group_faces = faces[group["i"], :n_vertex]
            group_coordinates = vertices[group_faces]
            geometries.append(shapely.polygons(group_coordinates))
        geometry = np.concatenate(geometries)
    return gpd.GeoDataFrame(geometry=geometry)


class Grouper:
    """
    Wrapper around pd.DataFrame().groupby().

    Group by ``a_index``, then iterates over the groups, returning a single
    value of ``a`` and potentially multiple values for ``b``.
    """

    def __init__(
        self,
        a,
        a_index,
        b,
        b_index,
    ):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.grouped = iter(
            pd.DataFrame(data={"a": a_index, "b": b_index}).groupby("a")
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            a_index, group = next(self.grouped)
            b_index = group["b"]
            return self.a[a_index], self.b[b_index]
        except StopIteration:
            raise StopIteration
