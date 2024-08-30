from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pandamesh.common import (
    BoolArray,
    GeometryArray,
    IntArray,
    flatten_geometries,
    flatten_geometry,
)
from pandamesh.snapping import snap_nodes


def collect_exteriors(geometry: GeometryArray) -> GeometryArray:
    return shapely.get_exterior_ring(geometry)


def collect_interiors(geometry: GeometryArray) -> GeometryArray:
    interiors = []
    for geom in geometry:
        interiors.extend(geom.interiors)
    return np.asarray(interiors)


def filter_holes_and_assign_values(
    polygons: GeometryArray,
    index_original: IntArray,
    index_union: IntArray,
    indexer: np.ndarray,
    ascending: bool,
) -> Tuple[BoolArray, np.ndarray]:
    df = pd.DataFrame(
        data={
            "union": index_union,
            "original": index_original,
            "indexer": indexer[index_original],
        }
    )
    result = (
        df.sort_values("indexer", ascending=ascending)
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
    ascending: bool,
) -> Tuple[GeometryArray, IntArray]:
    # new is the partioned unary union of old. Every linear ring will be a
    # polygon, this includes holes. Grab a point inside of new, then find out
    # where they are located (in which polygon) or even outside (in a hole).
    sampling_points = [polygon.representative_point() for polygon in new]
    tree = STRtree(old)
    index_union, index_original = tree.query(sampling_points, predicate="within")
    # In case of overlaps, there are multiple answers for each union polygon.
    new_polygons, new_indexer = filter_holes_and_assign_values(
        polygons=new,
        index_original=index_original,
        index_union=index_union,
        indexer=indexer,
        ascending=ascending,
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


def merge_polygons(geometry: GeometryArray, grid_size) -> GeometryArray:
    merged = shapely.unary_union(geometry, grid_size=grid_size)
    return np.asarray(flatten_geometry(merged))


def separate(
    geometry: GeometryArray,
) -> Tuple[GeometryArray, GeometryArray, GeometryArray]:
    type_id = shapely.get_type_id(geometry)
    acceptable = {"Point", "LineString", "LinearRing", "Polygon"}
    if not np.isin(type_id, (0, 1, 2, 3)).all():
        raise TypeError(
            f"Geometry should be one of {acceptable}. "
            "Call geopandas.GeoDataFrame.explode() to explode multi-part "
            "geometries into multiple single geometries."
        )
    return (
        geometry[type_id == 3],
        geometry[(type_id == 1) | (type_id == 2)],
        geometry[type_id == 0],
    )


class Preprocessor:
    """
    Utilities to clean-up geometries before meshing.

    Each processing method return a new Preprocessor instance. This enables
    method chaining.

    Note: many methods require exhaustive checking of geometries. Processing
    may be slow if your geometry contains thousands of geometries or if it has
    excessive detail.

    Parameters
    ----------
    geometry: np.ndarray of shapely geometries of length N.
        Should be points, linestrings, or polygons.
    values: np.ndarray of length N, optional
        Values associated with the geometries (e.g. cell sizes for meshing).
    grid_size: float, optional
        Forwarded to shapely operations. If grid_size is nonzero, input
        coordinates will be snapped to a precision grid of that size and
        resulting coordinates will be snapped to that same grid for shapely
        operations such as ``intersection, difference, union``, etc. If 0, the
        operations will use double precision coordinates.

    Examples
    --------
    This class allows for method chaining to flexibly combine pre-processing
    operations:

    >>> processed = (
    >>>     Preprocessor(geometry=gdf.geometry)
    >>>     .interpolate_lines_to_points(distance=5.0)
    >>>     .snap_points(distance=2.0)
    >>>     .unify_polygons()
    >>>     .merge_polygons()
    >>>     .clip_points(distance=5.0)
    >>>     .to_geodataframe()
    >>> )

    An intermediate result can be checked and inspect by calling ``.to_geodataframe()``
    at any time.

    >>> check = Preprocessor(geometry=gdf.geometry).interpolate_lines_to_points(5.0).to_geodataframe()
    """

    def __init__(
        self,
        geometry=None,
        values=None,
        grid_size=None,
    ):
        geometry = np.asarray(geometry)
        self.polygons, self.lines, self.points = separate(geometry)
        if values is not None:
            values = np.asarray(values)
            v_shape = values.shape
            g_shape = geometry.shape
            if v_shape != g_shape:
                raise ValueError(
                    f"geometry and values shape mismatch: {g_shape} versus "
                    f"{v_shape}"
                )
        self.values = values
        self.grid_size = grid_size
        self._set_indexers()

    def _set_indexers(self):
        n_polys = len(self.polygons)
        n_lines = len(self.lines)
        n_point = len(self.points)
        if self.values is not None:
            self.values, indexer = np.unique(self.values, return_inverse=True)
            self.polygon_indexer, self.line_indexer, self.point_indexer = np.split(
                indexer, [n_polys, n_polys + n_lines]
            )
        else:
            self.polygon_indexer = np.arange(0, n_polys)
            self.line_indexer = np.arange(n_polys, n_polys + n_lines)
            self.point_indexer = np.arange(
                n_polys + n_lines, n_polys + n_lines + n_point
            )
        return

    def _copy_with(
        self,
        values=None,
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
        new_instance = object.__new__(self.__class__)
        kwargs = {
            "values": values,
            "polygons": polygons,
            "lines": lines,
            "points": points,
            "polygon_indexer": polygon_indexer,
            "line_indexer": line_indexer,
            "point_indexer": point_indexer,
            "grid_size": None,
        }
        for k, v in kwargs.items():
            if v is None:
                v = getattr(self, k)
            setattr(new_instance, k, v)
        return new_instance

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Return the processed geometries as a ``geopandas.GeoDataFrame``.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            Contains columns geometry and indexer. The indexer column can be
            used to (re-)associate the geometry with the original values.
            If ``values`` were provided at initialization, a values column will
            be added to the geodataframe.
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
        data = {"indexer": indexer}
        if self.values is not None:
            data["values"] = self.values[indexer]
        return gpd.GeoDataFrame(
            data=data,
            geometry=geometry,
        )

    def merge_polygons(self) -> "Preprocessor":
        """
        Merge polygons with the same value for indexer (the same value if
        ``values`` was provided at initialization).

        Returns
        -------
        processed: pandamesh.Preprocessor
        """
        polygon_index = np.arange(len(self.polygons))
        df = pd.DataFrame(
            data={"polygon_index": polygon_index, "indexer": self.polygon_indexer}
        )
        merged_index = []
        merged = []
        for value, group in df.groupby("indexer"):
            group_polygons = self.polygons[group["polygon_index"]]
            merged_geometry = merge_polygons(group_polygons, grid_size=self.grid_size)
            merged.extend(merged_geometry)
            merged_index.extend([value] * len(merged_geometry))

        return self._copy_with(polygons=merged, polygon_indexer=merged_index)

    def unify_polygons(self, first: bool = True) -> "Preprocessor":
        """
        Resolve polygon overlaps and intersections.

        In overview, this method takes the following steps:

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
           the lowest indexer value or the smallest value in case ``.values``
           was provided at initialization if ``first=True``; the highest
           indexer value or largest value is taken for ``first=False``.
        6. re-associate with the original indexer and discard hole polygons.

        Unify polygons may generate many neighboring sub-polygons with the same
        indexer value. The can be merged with ``.merge_polygons``.

        Parameters
        ----------
        first: bool, optional
            Which value or index to assign in case of polygon overlap.

            In case ``values`` were provided at initialization:

            * ``first=True``: take the smallest value among the overlapping
              polygons.
            * ``first=False``: take the largest value among the
              overlapping polygons.

            If not provided:

            * ``first=True``: take the first overlapping polygon of the input
              geometry.
            * ``first=False``: take the last overlapping polygon of the input
              geometry.

            See the examples.

        Returns
        -------
        processed: pandamesh.Preprocessor

        Examples
        --------
        Resolve overlapping polygons, assigning the smallest cell size values
        in case of overlap:

        >>> processed = (
        >>>     pandamesh.Preprocessor(geometry=gdf["geometry"], values=gdf["cellsize"])
        >>>     .unify_polygons()
        >>>     .merge_polygons(first=True)
        >>>     .to_geodataframe()
        >>> )

        Assign the largest cell size value instead:

        >>> processed = (
        >>>     pandamesh.Preprocessor(geometry=gdf["geometry"], values=gdf["cellsize"])
        >>>     .unify_polygons()
        >>>     .merge_polygons(first=False)
        >>>     .to_geodataframe()
        >>> )

        Alternatively, to control the result of the merging with values, we can
        sort the ``gdf`` prior to processing.

        >>> sorted_gdf = gdf.sort_values(["a", "b"])
        >>> processor = pandamesh.Preprocessor(sorted_gdf["geometry"])

        Afterwards, the returned indexer can be used to fetch the data associated
        with the merged results.

        >>> out = processor.to_geodataframe()
        >>> processed = geopandas.GeoDataFrame(
        >>>     data=sorted_gdf.iloc[out["indexer"]].loc[["a", "b"]],
        >>>     geometry=out["geometry"],
        >>> )
        """
        rings = np.concatenate(
            (
                collect_exteriors(self.polygons),
                collect_interiors(self.polygons),
                self.lines,
            )
        )
        union = shapely.unary_union(rings, grid_size=self.grid_size)
        polygonized_union = flatten_geometries(list(shapely.polygonize([union]).geoms))
        new_polygons, new_indexer = locate_polygons(
            new=polygonized_union,
            old=self.polygons,
            indexer=self.polygon_indexer,
            ascending=first,
        )
        return self._copy_with(polygons=new_polygons, polygon_indexer=new_indexer)

    def clip_lines(self, distance=0.0) -> "Preprocessor":
        """
        Remove line segments that are outside or that are near polygon
        segments.

        Returns
        -------
        processed: pandamesh.Preprocessor
        """

        if len(self.lines) == 0:
            return self._copy_with()

        # Discard line elements outside of the polygons.
        all_polygons = shapely.unary_union(self.polygons, grid_size=self.grid_size)
        lines_inside = flatten_geometries(
            shapely.intersection(self.lines, all_polygons, grid_size=self.grid_size)
        )

        if distance > 0:
            # Remove lines near to polygon boundaries.
            rings = np.concatenate(
                (
                    collect_exteriors(self.polygons),
                    collect_interiors(self.polygons),
                )
            )
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
        return self._copy_with(lines=new_lines, line_indexer=new_indexer)

    def unify_lines(self) -> "Preprocessor":
        """
        Ensure intersections between lines are present.

        Returns
        -------
        processed: pandamesh.Preprocessor
        """
        if len(self.lines) == 0:
            return self._copy_with()

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
        return self._copy_with(
            lines=lines_union, line_indexer=self.line_indexer[index_original]
        )

    def clip_points(self, distance: float = 0.0) -> "Preprocessor":
        """
        Remove points that are outside of a polygon or near line or
        polygon segments.

        Parameters
        ----------
        distance: float, optional
            Minimum distance to a line or polygon segment.

        Returns
        -------
        processed: pandamesh.Preprocessor
        """
        if len(self.points) == 0:
            return self._copy_with()

        # Check whether points are inside.
        tree = STRtree(self.polygons)
        inside, _ = tree.query(self.points, predicate="within")
        new_points = self.points[inside]
        new_indexer = self.point_indexer[inside]

        # Check whether points aren't too near.
        rings = np.concatenate(
            (
                collect_exteriors(self.polygons),
                collect_interiors(self.polygons),
                self.lines,
            )
        )
        tree = STRtree(geoms=rings)
        too_near, _ = tree.query(new_points, predicate="dwithin", distance=distance)
        keep = np.full(len(new_points), True)
        keep[too_near] = False
        return self._copy_with(points=new_points[keep], point_indexer=new_indexer[keep])

    def interpolate_lines_to_points(
        self, distance: Optional[float] = None, values_as_distance: bool = False
    ) -> "Preprocessor":
        """
        Convert lines into points.

        This method adds vertices if needed, but it does not discard vertices
        that are located close together: ``distance`` is the maximum distance,
        not a minimum.

        Parameters
        ----------
        distance: float, optional
            Additional vertices will be added so that all line segments are no
            longer than this value. Must be greater than 0.
        values_as_distance: bool, optional
            If true, ignores the value of ``distance`` and uses ``values``
            provided during initialization as the distance instead. Errors if
            no values have been provided.

        Returns
        -------
        processed: pandamesh.Preprocessor
        """
        if len(self.lines) == 0:
            return self._copy_with()

        if values_as_distance:
            distance = self.values[self.line_indexer]
        elif distance is None:
            raise ValueError(
                "If values_as_distance is False, distance must be provided."
            )

        segmentized = shapely.segmentize(self.lines, distance)
        new_points_xy, index = shapely.get_coordinates(segmentized, return_index=True)
        return self._copy_with(
            points=np.concatenate((shapely.points(new_points_xy), self.points)),
            point_indexer=np.concatenate(
                (self.point_indexer, self.line_indexer[index])
            ),
            lines=np.empty(0, dtype=object),
            line_indexer=np.empty(0, dtype=int),
        )

    def snap_points(self, distance: float) -> "Preprocessor":
        """
        Snap points together that are within tolerance of each other.

        Will use Numba to accelerate the snapping if it is installed. This may
        be significantly faster in case of snapping a large (>10 000) number of points.

        Returns
        -------
        processed: pandamesh.Preprocessor
        """
        if len(self.points) == 0:
            return self._copy_with()
        index = snap_nodes(shapely.get_coordinates(self.points), distance)
        return self._copy_with(
            points=self.points[index],
            point_indexer=self.point_indexer[index],
        )
