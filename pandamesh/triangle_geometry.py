from typing import Tuple

import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry as sg

from .common import FloatArray, IntArray, flatten


def add_linestrings(linestrings: gpd.GeoSeries) -> Tuple[FloatArray, IntArray]:
    if len(linestrings) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    geometry = linestrings.geometry.values
    n_vertex = shapely.get_num_coordinates(geometry)
    vertices, index = shapely.get_coordinates(geometry, return_index=True)

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
        vertices[:, 0] = points.geometry.x
        vertices[:, 1] = points.geometry.y
    return vertices


def polygon_holes(
    polygons: gpd.GeoDataFrame,
):
    """
    Triangle recognizes holes as a point contained by segments.
    """
    inner_rings = gpd.GeoSeries(flatten(polygons.interiors))
    interiors = gpd.GeoDataFrame(geometry=[sg.Polygon(ring) for ring in inner_rings])
    points = interiors.representative_point()
    # Filter the points, if the point can be found in a polygon, it's located
    # in a refinement zone.
    points_inside = polygons.sjoin(
        gpd.GeoDataFrame(geometry=points), predicate="contains"
    )
    keep = np.full(len(points), True)
    keep[points_inside["index_right"].values.astype(int)] = False
    points = points[keep]

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
