import geopandas as gpd
import numpy as np
import shapely.geometry as sg

import pandamesh as pm
from pandamesh.triangle_mesher import add_linestrings, add_polygons

gpd.options.use_pygeos = False


outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
donut = sg.Polygon(outer, holes=[inner])


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

    series = gpd.GeoSeries(data=[donut])
    vertices, segments = add_linestrings(series)
    expected = np.unique(
        np.concatenate([outer_coords, inner_coords]),
        axis=0,
    )
    # TODO
    expected_segments = [
        [0, 6],
        [6, 7],
        [7, 1],
        [1, 0],
        [0, 2],
        [2, 4],
        [4, 5],
        [5, 3],
        [3, 2],
    ]
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
