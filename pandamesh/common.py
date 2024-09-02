from __future__ import annotations

import functools
import operator
import warnings
from typing import Any, Sequence, Set, Tuple

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
    are_simple = features.is_simple
    n_complex = (~are_simple).sum()
    if n_complex > 0:
        raise ValueError(
            f"{n_complex} cases of complex {feature_type} detected: these "
            " features contain self intersections"
        )
    return


def check_polygons(polygons: gpd.GeoSeries) -> None:
    check_features(polygons, "polygon")
    check_intersection(polygons, "polygon")
    return


def check_lines(lines: gpd.GeoSeries) -> None:
    check_features(lines, "lines")
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
    tolsquared = tolerance * tolerance
    unresolved = (distance_i > tolsquared) | (distance_j > tolsquared)
    return intersections[unresolved]


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


def _separate(
    geometry: GeometryArray,
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
    geometry: GeometryArray,
) -> Tuple[GeometryArray, GeometryArray, GeometryArray]:
    is_polygon, is_line, is_point = _separate(geometry)
    return (
        geometry[is_polygon],
        geometry[is_line],
        geometry[is_point],
    )


def separate_geodataframe(
    gdf: gpd.GeoDataFrame,
    intersecting_edges: str,
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

    check_polygons(polygons.geometry)
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
