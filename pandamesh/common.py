from __future__ import annotations

import functools
import operator
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
    index_a, index_b = intersecting_features(features, feature_type)
    n_overlap = len(index_a)
    if n_overlap > 0:
        message = "\n".join([f"{a} with {b}" for a, b in zip(index_a, index_b)])
        raise ValueError(
            f"{n_overlap} cases of intersecting {feature_type} detected:\n{message}"
        )


def check_features(features: gpd.GeoSeries, feature_type) -> None:
    """
    Features should:

        * be simple: no self-intersection
        * not intersect with other features

    """
    # Check valid
    are_simple = features.is_simple
    n_complex = (~are_simple).sum()
    if n_complex > 0:
        raise ValueError(
            f"{n_complex} cases of complex {feature_type} detected: these "
            " features contain self intersections"
        )

    if len(features) <= 1:
        return

    check_intersection(features, feature_type)
    return


def check_polygons(polygons: gpd.GeoSeries) -> None:
    check_features(polygons, "polygon")


def check_linestrings(
    linestrings: gpd.GeoSeries,
    polygons: gpd.GeoSeries,
) -> None:
    """Check whether linestrings are fully contained in a single polygon."""
    check_features(linestrings, "linestring")

    intersects = gpd.GeoDataFrame(geometry=linestrings).sjoin(
        df=gpd.GeoDataFrame(geometry=polygons),
        predicate="within",
    )
    n_diff = len(linestrings) - len(intersects)
    if n_diff != 0:
        raise ValueError(
            "The same linestring detected in multiple polygons or "
            "linestring detected outside of any polygon; "
            "a linestring must be fully contained by a single polygon."
        )

    return


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
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    is_polygon, is_line, is_point = _separate(gdf["geometry"])
    polygons = gdf.loc[is_polygon].copy()
    linestrings = gdf.loc[is_line].copy()
    points = gdf.loc[is_point].copy()
    for df in (polygons, linestrings, points):
        df["cellsize"] = df["cellsize"].astype(float)
        df.crs = None

    check_polygons(polygons.geometry)
    # TODO: do a better check, on segments instead of the entire linestring.
    # check_linestrings(linestrings.geometry, polygons.geometry)
    check_points(points.geometry, polygons.geometry)

    return polygons, linestrings, points


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
