from enum import Enum
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pygeos
import shapely.geometry as sg
import triangle

from .common import FloatArray, IntArray, flatten, invalid_option, separate


def add_linestrings(linestrings: gpd.GeoSeries) -> Tuple[FloatArray, IntArray]:
    if len(linestrings) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    geometry = linestrings.geometry.values
    geometry = pygeos.from_shapely(geometry)
    n_vertex = pygeos.get_num_coordinates(geometry)
    vertices, index = pygeos.get_coordinates(geometry, return_index=True)

    vertex_numbers = np.arange(n_vertex.sum())
    segments = np.empty((n_vertex.sum() - 1, 2), dtype=np.int32)
    segments[:, 0] = vertex_numbers[:-1]
    segments[:, 1] = vertex_numbers[1:]
    keep = np.diff(index) == 0
    segments = segments[keep]

    # If the geometry is closed (LinearRings), the final vertex of every
    # feature is discarded, since the repeats will segfault Triangle.
    vertices, inverse = np.unique(vertices, return_inverse=True, axis=0)
    segments = inverse[segments]

    return vertices, segments


def add_polygons(
    polygons: gpd.GeoDataFrame,
) -> Tuple[FloatArray, IntArray, FloatArray]:
    is_region = polygons["cellsize"].notnull()
    n_region = is_region.sum()
    regions = np.empty((n_region, 4), dtype=np.float64)
    region_points = polygons[is_region].representative_point()
    regions[:, 0] = region_points.x
    regions[:, 1] = region_points.y
    regions[:, 2] = np.arange(n_region)
    cellsize = polygons[is_region]["cellsize"].values
    regions[:, 3] = 0.5 * cellsize * cellsize

    boundary = polygons.boundary.explode(index_parts=True).geometry
    vertices, segments = add_linestrings(boundary)
    return vertices, segments, regions


def add_points(points: gpd.GeoDataFrame) -> FloatArray:
    vertices = np.empty((len(points), 2), dtype=np.float64)
    if len(points) > 0:
        vertices[:, 0] = points.x
        vertices[:, 1] = points.y
    return vertices


def polygon_holes(
    polygons: gpd.GeoDataFrame,
):
    """
    Triangle recognizes holes as a point contained by segments.
    """
    inner_rings = gpd.GeoSeries(flatten(polygons.interiors))
    interiors = gpd.GeoDataFrame(geometry=[sg.asPolygon(ring) for ring in inner_rings])
    points = interiors.representative_point()
    if len(points) > 0:
        return add_points(points)
    else:
        return None


def collect_geometry(
    polygons: gpd.GeoDataFrame, linestrings: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> Tuple[FloatArray, IntArray, FloatArray]:
    if len(polygons) == 0:
        raise ValueError("No polygons provided")
    polygon_vertices, polygon_segments, regions = add_polygons(polygons)
    linestring_vertices, linestring_segments = add_linestrings(linestrings.geometry)
    point_vertices = add_points(points)
    linestring_segments += polygon_vertices.shape[0]
    vertices = np.concatenate([polygon_vertices, linestring_vertices, point_vertices])
    segments = np.concatenate([polygon_segments, linestring_segments])
    return vertices, segments, regions


class DelaunayAlgorithm(Enum):
    DIVIDE_AND_CONQUER = ""
    INCREMENTAL = "i"
    SWEEPLINE = "F"


class TriangleMesher:
    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        polygons, linestrings, points = separate(gdf)
        self.vertices, self.segments, self.regions = collect_geometry(
            polygons, linestrings, points
        )
        self.holes = polygon_holes(polygons)

        # Set default values for meshing parameters
        self.minimum_angle = 20.0
        self.conforming_delaunay = True
        self.suppress_exact_arithmetic = False
        self.maximum_steiner_points = None
        self.delaunay_algorithm = DelaunayAlgorithm.DIVIDE_AND_CONQUER
        self.consistency_check = False

    @property
    def minimum_angle(self) -> float:
        return self._minimum_angle

    @minimum_angle.setter
    def minimum_angle(self, value: float):
        if not isinstance(value, float):
            raise TypeError("minimum angle must be a float")
        if value >= 34.0 or value <= 0.0:
            raise ValueError("minimum_angle should fall in the interval: (0.0, 34.0)")
        self._minimum_angle = value

    @property
    def conforming_delaunay(self) -> bool:
        return self._conforming_delaunay

    @conforming_delaunay.setter
    def conforming_delaunay(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("conforming_delaunay must be a bool")
        self._conforming_delaunay = value

    @property
    def suppress_exact_arithmetic(self) -> bool:
        return self._suppress_exact_arithmetic

    @suppress_exact_arithmetic.setter
    def suppress_exact_arithmetic(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("suppress_exact_arithmetic must be a bool")
        self._suppress_exact_arithmetic = value

    @property
    def maximum_steiner_points(self) -> int:
        return self._maximum_steiner_points

    @maximum_steiner_points.setter
    def maximum_steiner_points(self, value: Union[int, None]) -> None:
        if not isinstance(value, (int, type(None))):
            raise TypeError("maximum_steiner_points must be an int or None")
        self._maximum_steiner_points = value

    @property
    def delaunay_algorithm(self) -> DelaunayAlgorithm:
        return self._delaunay_algorithm

    @delaunay_algorithm.setter
    def delaunay_algorithm(self, value: DelaunayAlgorithm) -> None:
        if value not in DelaunayAlgorithm:
            raise ValueError(invalid_option(value, DelaunayAlgorithm))
        self._delaunay_algorithm = value

    @property
    def consistency_check(self) -> bool:
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self, value: bool) -> None:
        if not isinstance(value, int):
            raise TypeError("consistency_check must be a bool")
        self._consistency_check = value

    def generate(self) -> Tuple[FloatArray, IntArray]:
        options = (
            "p"
            f"q{self._minimum_angle}"
            "a"
            f"{'D' if self._conforming_delaunay else ''}"
            f"{'X' if self._suppress_exact_arithmetic else ''}"
            f"{'S' + str(self._maximum_steiner_points) if self._maximum_steiner_points is not None else ''}"
            f"{self._delaunay_algorithm.value}"
            f"{'C' if self.consistency_check else ''}"
        )

        tri = {"vertices": self.vertices, "segments": self.segments}
        if self.holes is not None:
            tri["holes"] = self.holes
        if len(self.regions) > 0:
            tri["regions"] = self.regions

        result = triangle.triangulate(tri=tri, opts=options)
        return result["vertices"], result["triangles"]
