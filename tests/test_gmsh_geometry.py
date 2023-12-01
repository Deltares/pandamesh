import geopandas as gpd
import numpy as np
import shapely.geometry as sg

from pandamesh import gmsh_geometry as gg
from pandamesh.gmsh_mesher import gmsh_env

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
line = sg.LineString(line_coords)
polygon = sg.Polygon(outer_coords)
donut = sg.Polygon(outer_coords, holes=[inner_coords])
refined = sg.Polygon(inner_coords)

y = np.arange(0.5, 10.0, 0.5)
x = np.full(y.size, 1.0)
points_embed = gpd.points_from_xy(x, y)
line_embed = sg.LineString(
    [
        [9.0, 2.0],
        [9.0, 8.0],
    ]
)
polygons = gpd.GeoDataFrame({"cellsize": [1.0], "__polygon_id": [1]}, geometry=[donut])


def test_polygon_info():
    info, vertices, cellsizes, index = gg.polygon_info(polygon, 1.0, 0, 0)
    expected_info = gg.PolygonInfo(0, 4, [], [], 0)
    assert np.allclose(vertices, outer_coords)
    assert info == expected_info
    assert np.allclose(cellsizes, 1.0)
    assert index == 4

    info, vertices, cellsizes, index = gg.polygon_info(donut, 1.0, 0, 0)
    expected_info = gg.PolygonInfo(0, 4, [4], [4], 0)
    assert np.allclose(vertices, [outer_coords, inner_coords])
    assert info == expected_info
    assert np.allclose(cellsizes, 1.0)
    assert index == 8


def test_linestring_info():
    info, vertices, cellsizes, index = gg.linestring_info(line, 1.0, 0, 0)
    expected_info = gg.LineStringInfo(0, 2, 0)
    assert np.allclose(vertices, line_coords)
    assert info == expected_info
    assert np.allclose(cellsizes, 1.0)
    assert index == 2


def test_add_vertices():
    with gmsh_env():
        gg.add_vertices(
            [(0.0, 0.0), (1.0, 1.0)],
            [1.0, 1.0],
            [0, 1],
        )


def test_add_linestrings():
    info = gg.LineStringInfo(0, 2, 0)
    with gmsh_env():
        gg.add_vertices(
            [(0.0, 0.0), (1.0, 1.0)],
            [1.0, 1.0],
            [0, 1],
        )
        gg.add_linestrings([info], [0, 1])


def test_add_curve_loop():
    with gmsh_env():
        gg.add_vertices(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            [1.0, 1.0, 1.0, 1.0],
            [0, 1, 2, 3],
        )
        curve_loop_tag = gg.add_curve_loop([0, 1, 2, 3])
    assert curve_loop_tag == 1


def test_add_polygons():
    info = gg.PolygonInfo(0, 4, [4], [4], 0)
    vertices = np.vstack([inner_coords, outer_coords])
    cellsizes = np.full(vertices.size, 1.0)
    tags = np.arange(vertices.size)
    with gmsh_env():
        gg.add_vertices(vertices, cellsizes, tags)
        loop_tags, plane_tags = gg.add_polygons([info], tags)
    assert loop_tags == [1, 2]
    assert plane_tags == [0]


def test_add_points():
    x = np.arange(0.5, 10.0, 0.5)
    y = np.full(x.size, 1.0)
    ids = np.arange(x.size) + 1
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
    gdf["__polygon_id"] = ids
    gdf["cellsize"] = 1.0
    with gmsh_env():
        indices, embedded_in = gg.add_points(gdf)
    assert np.array_equal(indices, ids)
    assert np.array_equal(embedded_in, ids)


def test_collect_polygons():
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    gdf["cellsize"] = 1.0
    gdf["__polygon_id"] = 1
    index, vertices, cellsizes, features = gg.collect_polygons(gdf, 0)
    assert index == 4
    assert np.allclose(vertices, outer_coords)
    assert np.allclose(cellsizes, 1.0)
    assert features == [gg.PolygonInfo(0, 4, [], [], 1)]


def test_collect_linestrings():
    gdf = gpd.GeoDataFrame(geometry=[line])
    gdf["cellsize"] = 1.0
    gdf["__polygon_id"] = 1
    index, vertices, cellsizes, features = gg.collect_linestrings(gdf, 0)
    assert index == 2
    assert np.allclose(vertices, line_coords)
    assert np.allclose(cellsizes, 1.0)
    assert features == [gg.LineStringInfo(0, 2, 1)]


def test_collect_points():
    x = np.arange(0.5, 10.0, 0.5)
    y = np.full(x.size, 1.0)
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
    xy = gg.collect_points(gdf)
    assert np.allclose(xy, np.column_stack([x, y]))


def test_embed_where_linestring():
    line_gdf = gpd.GeoDataFrame({"cellsize": [0.5]}, geometry=[line_embed])
    actual = gg.embed_where(line_gdf, polygons)
    assert np.allclose(actual["cellsize"], 0.5)
    assert np.allclose(actual["__polygon_id"], 1)
    assert actual.geometry.iloc[0] == line_embed


def test_embed_where_points():
    points_gdf = gpd.GeoDataFrame(geometry=points_embed)
    points_gdf["cellsize"] = 0.25
    actual = gg.embed_where(points_gdf, polygons)
    assert np.allclose(actual["cellsize"], 0.25)
    assert np.allclose(actual["__polygon_id"], 1)
    assert (actual.geometry.to_numpy() == points_embed).all()


def test_add_geometry():
    line_gdf = gpd.GeoDataFrame({"cellsize": [0.5]}, geometry=[line_embed])
    points_gdf = gpd.GeoDataFrame(geometry=points_embed)
    points_gdf["cellsize"] = 0.25
    with gmsh_env():
        gg.add_geometry(polygons, line_gdf, points_gdf)


# TODO: add tests for fields functionality
