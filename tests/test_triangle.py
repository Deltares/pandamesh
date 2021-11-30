import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg

import pandamesh as pm
from pandamesh.triangle_mesher import (
    add_linestrings,
    add_points,
    add_polygons,
    polygon_holes,
)

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
donut = sg.Polygon(outer, holes=[inner])
refined = sg.Polygon(inner_coords)


def area(vertices, triangles):
    """
    Compute the area of every triangle in the mesh.
    (Helper for these tests.)
    """
    coords = vertices[triangles]
    u = coords[:, 1] - coords[:, 0]
    v = coords[:, 2] - coords[:, 0]
    return 0.5 * np.abs(np.cross(u, v))


def test_add_linestrings():
    series = gpd.GeoSeries(data=[line])
    vertices, segments = add_linestrings(series)
    expected = np.unique(line_coords, axis=0)
    expected_segments = np.array([[0, 1]])
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    series = gpd.GeoSeries(data=[inner])
    vertices, segments = add_linestrings(series)
    expected = np.unique(inner_coords, axis=0)
    expected_segments = np.array(
        [
            [0, 2],
            [2, 3],
            [3, 1],
            [1, 0],
        ]
    )
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    series = gpd.GeoSeries(data=[outer])
    vertices, segments = add_linestrings(series)
    expected = np.unique(outer_coords, axis=0)
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    # Empty should work too
    series = gpd.GeoSeries(data=[])
    _, _ = add_linestrings(series)


def test_add_polygons():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    cellsize = 0.5
    gdf["cellsize"] = cellsize
    vertices, segments, regions = add_polygons(gdf)
    expected = np.unique(
        np.concatenate([outer_coords, inner_coords]),
        axis=0,
    )
    expected_segments = np.array(
        [
            [0, 6],
            [6, 7],
            [7, 1],
            [1, 0],
            [2, 4],
            [4, 5],
            [5, 3],
            [3, 2],
        ]
    )
    x, y = regions[0, :2]
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)
    assert regions[0, 2] == 0
    assert regions[0, 3] == 0.5 * cellsize ** 2
    assert sg.Point(x, y).within(donut)


def test_add_points():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    vertices = add_points(gdf)
    assert np.allclose(vertices, xy)


def test_polygon_holes():
    polygon = sg.Polygon(outer)
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    assert polygon_holes(gdf) is None

    gdf = gpd.GeoDataFrame(geometry=[donut])
    assert len(polygon_holes(gdf)) == 1

    gdf = gpd.GeoDataFrame(geometry=[donut, refined])
    assert polygon_holes(gdf) is None


def test_triangle_basic():
    polygon = sg.Polygon(outer)
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    gdf["cellsize"] = 1.0
    mesher = pm.TriangleMesher(gdf)
    vertices, triangles = mesher.generate()
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, polygon.area)


def test_triangle_hole():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.TriangleMesher(gdf)
    vertices, triangles = mesher.generate()
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, donut.area)


def test_triangle_adjacent_donut():
    inner_coords2 = inner_coords.copy()
    outer_coords2 = outer_coords.copy()
    inner_coords2[:, 0] += 10.0
    outer_coords2[:, 0] += 10.0
    inner2 = sg.LinearRing(inner_coords2)
    outer2 = sg.LinearRing(outer_coords2)
    donut2 = sg.Polygon(outer2, holes=[inner2])

    gdf = gpd.GeoDataFrame(geometry=[donut, donut2])
    gdf["cellsize"] = [1.0, 0.5]
    mesher = pm.TriangleMesher(gdf)
    vertices, triangles = mesher.generate()
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)

    # With a line at y=8.0 and points in the left polygon, at y=2.0
    line1 = sg.LineString([(0.25, 8.0), (9.75, 8.0)])
    line2 = sg.LineString([(10.25, 8.0), (19.75, 8.0)])
    x = np.arange(0.25, 10.0, 0.25)
    y = np.full(x.size, 2.0)
    points = gpd.points_from_xy(x=x, y=y)
    gdf = gpd.GeoDataFrame(geometry=[donut, donut2, line1, line2, *points])
    gdf["cellsize"] = 1.0

    mesher = pm.TriangleMesher(gdf)
    vertices, triangles = mesher.generate()
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)


def test_triangle_properties():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.TriangleMesher(gdf)

    # Should be a float >=0, < 34.0
    with pytest.raises(TypeError):
        mesher.minimum_angle = 10
    with pytest.raises(ValueError):
        mesher.minimum_angle = 35.0

    # Set properties
    mesher.minimum_angle = 10.0
    mesher.conforming_delaunay = False
    mesher.suppress_exact_arithmetic = True
    mesher.maximum_steiner_points = 10
    mesher.delaunay_algorithm = pm.DelaunayAlgorithm.SWEEPLINE
    mesher.consistency_check = True

    # Check whether properties have been set properly
    assert mesher.minimum_angle == 10.0
    assert mesher.conforming_delaunay is False
    assert mesher.suppress_exact_arithmetic is True
    assert mesher.maximum_steiner_points == 10
    assert mesher.delaunay_algorithm == pm.DelaunayAlgorithm.SWEEPLINE
    assert mesher.consistency_check is True

    # Check whether the repr method works
    assert isinstance(mesher.__repr__(), str)
