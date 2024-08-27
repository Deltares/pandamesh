from typing import Any, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

GeometryArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


def collect_exteriors(geometry: GeometryArray) -> List:
    return list(shapely.get_exterior_ring(geometry))


def collect_interiors(geometry: GeometryArray) -> List:
    interiors = []
    for geom in geometry:
        interiors.extend(geom.interiors)
    return interiors


def flatten_geometry(geom):
    """Recursively flatten geometry collections."""
    if hasattr(geom, "geoms"):
        return [g for subgeom in geom.geoms for g in flatten_geometry(subgeom)]
    else:
        return [geom]


def flatten_geometries(geometries: List) -> GeometryArray:
    """Flatten geometry collections."""
    flattened = []
    for geom in geometries:
        flattened.extend(flatten_geometry(geom))
    return np.array(flattened)


def filter_holes_and_assign_values(
    polygons: GeometryArray,
    index_original: IntArray,
    index_union: IntArray,
    indexer: np.ndarray,
) -> Tuple[BoolArray, np.ndarray]:
    df = pd.DataFrame(
        data={
            "union": index_union,
            "original": index_original,
            "indexer": indexer[index_original],
        }
    )
    result = (
        df.sort_values("indexer", ascending=True)
        .groupby("union", as_index=False)
        .first()
    )
    maintain = np.full(len(polygons), False)
    maintain[df["union"]] = True
    return polygons[maintain], result["indexer"].to_numpy()


def locate_polygons(
    new: GeometryArray,
    old: GeometryArray,
    indexer: IntArray,
) -> Tuple[GeometryArray, IntArray]:
    sampling_points = [polygon.representative_point() for polygon in new]
    tree = STRtree(old)
    index_union, index_original = tree.query(sampling_points, predicate="within")
    new_polygons, new_indexer = filter_holes_and_assign_values(
        polygons=new,
        index_original=index_original,
        index_union=index_union,
        indexer=indexer,
    )
    return new_polygons, new_indexer


def locate_lines(
    new: GeometryArray,
    old: GeometryArray,
    indexer: IntArray,
) -> IntArray:
    sampling_points = shapely.line_interpolate_point(new, distance=0.5, normalized=True)
    tree = STRtree(old)
    index_old = tree.nearest(sampling_points)
    return indexer[index_old]


def intersect_lines_with_polygons(
    polygons: GeometryArray,
    lines: GeometryArray,
    polygon_index: IntArray,
    line_index: IntArray,
    grid_size: float,
) -> GeometryArray:
    grouped = pd.DataFrame(
        data={"polygon_index": polygon_index, "line_index": line_index}
    ).groupby("line_index")
    new_lines: List[Any] = []
    for group_index, group in grouped:
        line = lines[group_index]
        group_polygons = polygons[group["polygon_index"]]
        new_lines.extend(
            flatten_geometries(
                shapely.intersection(line, group_polygons, grid_size=grid_size)
            )
        )
    return np.asarray(new_lines)


def merge_polygons(geometry: GeometryArray, grid_size=None) -> GeometryArray:
    merged = shapely.unary_union(geometry, grid_size=grid_size)
    return np.asarray(flatten_geometry(merged))


def separate(
    geometry: GeometryArray,
) -> Tuple[GeometryArray, GeometryArray, GeometryArray]:
    type_id = shapely.get_type_id(geometry)
    acceptable = {"Point", "LineString", "LinearRing", "Polygon"}
    if not np.isin(type_id, (0, 1, 2, 3)).all():
        raise TypeError(f"Geometry should be one of {acceptable}")
    return (
        geometry[type_id == 3],
        geometry[(type_id == 1) | (type_id == 2)],
        geometry[type_id == 0],
    )


class Preprocessor:
    """
    Utilities to clean-up geometries before meshing.

    Processing methods return a new Preprocessor instance. This enables method
    chaining.

    Parameters
    ----------
    geometry: np.ndarray of shapely geometries, optional
    polygons: np.ndarray of shapely polygons, optional
    lines: np.ndarray of shapely lines, optional
    points: np.ndarray of shapely points, optional
    polygon_indexer: np.ndarray of ints, optional
    line_indexer: np.ndarray of ints, optional
    point_indexer: np.ndarray of ints, optional
    grid_size: float, optional
        If grid_size is nonzero, input coordinates will be snapped to a
        precision grid of that size and resulting coordinates will be snapped
        to that same grid. If 0, this operation will use double precision
        coordinates. Forwarded to shapely operations.
    """

    def __init__(
        self,
        geometry=None,
        values=None,
        polygons=None,
        lines=None,
        points=None,
        polygon_indexer=None,
        line_indexer=None,
        point_indexer=None,
        grid_size=None,
    ):
        if geometry is None:
            self.polygons = np.asarray(polygons)
            self.lines = np.asarray(lines)
            self.points = np.asarray(points)
        else:
            self.polygons, self.lines, self.points = separate(geometry)

        n_polys = len(self.polygons)
        n_lines = len(self.lines)
        n_point = len(self.points)
        if values is not None:
            _, indexer = np.unique(values, return_inverse=True)
            polygon_indexer, line_indexer, point_indexer = np.split(
                indexer, [n_polys, n_polys + n_lines]
            )

        if polygon_indexer is None:
            self.polygon_indexer = np.arange(0, n_polys)
        else:
            self.polygon_indexer = polygon_indexer

        if line_indexer is None:
            self.line_indexer = np.arange(n_polys, n_polys + n_lines)
        else:
            self.line_indexer = line_indexer

        if point_indexer is None:
            self.point_indexer = np.arange(
                n_polys + n_lines, n_polys + n_lines + n_point
            )
        else:
            self.point_indexer = point_indexer

        self.grid_size = grid_size
        return

    def to_gdf(self) -> gpd.GeoDataFrame:
        """
        Return the processed geometries as a ``geopandas.GeoDataFrame```.

        Returns
        -------
        gdf: gpd.GeoDataFrame
            Contains columns geometry and indexer. The indexer column can be
            used to (re-)associate the geometry with the original values.
        """
        geometry = np.concatenate(
            (
                self.polygons,
                self.lines,
                self.points,
            )
        )
        indexer = np.concatenate(
            (
                self.polygon_indexer,
                self.line_indexer,
                self.point_indexer,
            )
        )
        return gpd.GeoDataFrame(
            data={"indexer": indexer},
            geometry=geometry,
        )

    def merge_polygons(self) -> "Preprocessor":
        """Merge polygons with the same value for indexer."""
        index = np.arange(len(self.polygon_indexer))
        df = pd.DataFrame(data={"index": index, "indexer": self.polygon_indexer})
        merged_index = []
        merged = []
        for value, group in df.groupby("indexer"):
            indices = group["index"]
            geometry = merge_polygons(self.polygons[indices], grid_size=self.grid_size)
            merged.extend(geometry)
            merged_index.extend([value] * len(geometry))
        return self._copy(
            polygons=merged, polygon_indexer=self.polygon_indexer[merged_index]
        )

    def unify_polygons(self) -> "Preprocessor":
        """
        Resolve polygon overlaps and intersections.

        In overview:

        1. collect all linear rings (exterior and interior boundaries), as well
           as the linestrings.
        2. create a unary union of all the linework. This ensures intersections
           between lines are represented by a point on the lines.
        3. polygonize the union linework. This creates a polygon for each ring that
           is encountered, including holes.
        4. collect sampling points for the newly created polygons. Use these to
           locate in which polygon (or which hole!) the newly created polygon is
           located.
        5. In case of overlapping polygons, the sampling point may be present
           in more than one of the original polygons. We choose the one with
           the lowest indexer value.
        6. re-associate with the original indexer and discard hole polygons.

        Unify polygons may generate many neighboring sub-polygons with the same
        indexer value. The can be merged with ``.merge_polygons``.
        """
        rings = (
            collect_exteriors(self.polygons)
            + collect_interiors(self.polygons)
            + list(self.lines)
        )
        union = shapely.unary_union(rings, grid_size=self.grid_size)
        polygonized_union = flatten_geometries(list(shapely.polygonize([union]).geoms))
        new_polygons, new_indexer = locate_polygons(
            new=polygonized_union,
            old=self.polygons,
            indexer=self.polygon_indexer,
        )
        return self._copy(polygons=new_polygons, polygon_indexer=new_indexer)

    def clip_linestrings(self, distance=0.0) -> "Preprocessor":
        """
        Remove line segments that are outside or that are near polygon
        segments.
        """

        if len(self.lines) == 0:
            return self._copy()

        # Discard line elements outside of the polygons.
        all_polygons = shapely.unary_union(self.polygons, grid_size=self.grid_size)
        lines_inside = flatten_geometries(
            shapely.intersection(self.lines, all_polygons, grid_size=self.grid_size)
        )

        if distance > 0:
            # Remove lines near to polygon boundaries.
            rings = collect_exteriors(self.polygons) + collect_interiors(self.polygons)
            exclusion = shapely.unary_union(
                shapely.buffer(rings, distance=distance, cap_style="flat"),
                grid_size=self.grid_size,
            )
            new_lines = shapely.difference(
                lines_inside, exclusion, grid_size=self.grid_size
            )
            new_lines = new_lines[~shapely.is_empty(new_lines)]
        else:
            new_lines = lines_inside

        new_indexer = locate_lines(new_lines, self.lines, self.line_indexer)
        return self._copy(lines=new_lines, line_indexer=new_indexer)

    def unify_lines(self) -> "Preprocessor":
        """Ensure intersections between lines are present."""
        if len(self.lines) == 0:
            return self._copy()

        lines_union = np.asarray(
            flatten_geometry(shapely.unary_union(self.lines, grid_size=self.grid_size))
        )
        lines_union = lines_union[shapely.get_type_id(lines_union) == 1]
        # Sample halfway along the lines
        sampling_points = shapely.line_interpolate_point(
            lines_union, distance=0.5, normalized=True
        )
        tree = STRtree(self.lines)
        index_original = tree.nearest(sampling_points)
        return self._copy(
            lines=lines_union, line_indexer=self.line_indexer[index_original]
        )

    def clip_points(self, distance: Optional[float] = None) -> "Preprocessor":
        """
        Remove points that are outside of a polygon.

        Remove points that are near line or polygon segments.
        """
        if len(self.points) == 0:
            return self._copy()

        # Check whether points are inside.
        tree = STRtree(self.polygons)
        _, inside = tree.query(self.points, predicate="inside")
        new_points = self.points[inside]
        new_indexer = self.point_indexer[inside]

        # Check whether points aren't too near.
        rings = (
            collect_exteriors(self.polygons)
            + collect_interiors(self.polygons)
            + list(self.lines)
        )
        tree = STRtree(geoms=rings)
        _, too_near = tree.query(new_points, predicate="dwithin", distance=distance)
        keep = np.full(len(new_points), True)
        keep[too_near] = False

        new_points = new_points[keep]
        new_indexer = new_indexer[keep]
        return self._copy(points=new_points, point_indexer=new_indexer)

    def _copy(
        self,
        polygons=None,
        lines=None,
        points=None,
        polygon_indexer=None,
        line_indexer=None,
        point_indexer=None,
    ) -> "Preprocessor":
        """
        Create a copy except for the provided keywords.

        Used to return a new object easily to allow method chaining.
        """
        kwargs = {
            "polygons": polygons,
            "lines": lines,
            "points": points,
            "polygon_indexer": polygon_indexer,
            "line_indexer": line_indexer,
            "point_indexer": point_indexer,
            "grid_size": None,
        }
        kwargs = {
            k: v if v is not None else getattr(self, k) for k, v in kwargs.items()
        }
        return Preprocessor(**kwargs)
