import numpy as np
import shapely.geometry as sg

from pandamesh import gmsh_geometry as gg
from pandamesh.gmsh_mesher import gmsh_env

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
polygon = sg.Polygon(outer)
donut = sg.Polygon(outer, holes=[inner])
refined = sg.Polygon(inner_coords)


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
