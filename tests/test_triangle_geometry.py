import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry as sg

from pandamesh import triangle_geometry as tg

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
ring_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0), (3.0, 3.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
ring = sg.LineString(ring_coords)
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
    return 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def test_add_linestrings():
    series = gpd.GeoSeries(data=[line])
    vertices, segments = tg.add_linestrings(series)
    expected = np.unique(line_coords, axis=0)
    expected_segments = np.array([[0, 1]])
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    series = gpd.GeoSeries(data=[inner])
    vertices, segments = tg.add_linestrings(series)
    vertices, segments = tg.unique_vertices_and_segments(vertices, segments)
    expected = np.unique(inner_coords, axis=0)
    expected_segments = np.array(
        [
            [0, 2],
            [1, 0],
            [2, 3],
            [3, 1],
        ]
    )
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    series = gpd.GeoSeries(data=[outer])
    vertices, segments = tg.add_linestrings(series)
    vertices, segments = tg.unique_vertices_and_segments(vertices, segments)
    expected = np.unique(outer_coords, axis=0)
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)

    # Empty should work too
    series = gpd.GeoSeries(data=[])
    _, _ = tg.add_linestrings(series)


def test_add_polygons():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    cellsize = 0.5
    gdf["cellsize"] = cellsize
    vertices, segments, regions = tg.add_polygons(gdf)
    vertices, segments = tg.unique_vertices_and_segments(vertices, segments)
    expected = np.unique(
        np.concatenate([outer_coords, inner_coords]),
        axis=0,
    )
    expected_segments = np.array(
        [
            [0, 6],
            [1, 0],
            [2, 4],
            [3, 2],
            [4, 5],
            [5, 3],
            [6, 7],
            [7, 1],
        ]
    )
    x, y = regions[0, :2]
    assert np.allclose(vertices, expected)
    assert np.array_equal(segments, expected_segments)
    assert regions[0, 2] == 0
    assert regions[0, 3] == 0.5 * cellsize**2
    assert sg.Point(x, y).within(donut)


def test_add_points():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    vertices = tg.add_points(gdf)
    assert np.allclose(vertices, xy)


def test_polygon_holes():
    polygon = sg.Polygon(outer)
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    assert tg.polygon_holes(gdf) is None

    gdf = gpd.GeoDataFrame(geometry=[donut])
    assert len(tg.polygon_holes(gdf)) == 1

    gdf = gpd.GeoDataFrame(geometry=[donut, refined])
    assert tg.polygon_holes(gdf) is None


def test_convert_ring_linestring():
    # The linestring forms a ring within the outer polygon. During the
    # conversion, the ring should be converted to a second polygon, and a hole
    # should be made in the first polygon.
    polygon = sg.Polygon(shell=outer_coords)
    polygons = gpd.GeoDataFrame(geometry=[polygon])
    polygons["cellsize"] = 1.0
    linestrings = gpd.GeoDataFrame(geometry=[ring])

    new_polygons = tg.convert_linestring_rings(polygons, linestrings)
    assert isinstance(new_polygons, gpd.GeoDataFrame)
    assert np.allclose(new_polygons["cellsize"], 1.0)
    assert np.allclose(new_polygons.area, [84.0, 16.0])


def test_convert_ring_linestring__hole():
    # This second case has a hole inside of the linestring ring. The hole
    # should be preserved.
    inner = [
        [4.0, 6.0],
        [6.0, 6.0],
        [6.0, 4.0],
        [4.0, 4.0],
    ]
    polygon = sg.Polygon(shell=outer_coords, holes=[inner])
    polygons = gpd.GeoDataFrame(geometry=[polygon])
    polygons["cellsize"] = 1.0
    linestrings = gpd.GeoDataFrame(geometry=[ring])

    new_polygons = tg.convert_linestring_rings(polygons, linestrings)
    assert isinstance(new_polygons, gpd.GeoDataFrame)
    assert np.allclose(new_polygons["cellsize"], 1.0)
    assert np.allclose(new_polygons.area, [84.0, 12.0])


def test_convert_ring_linestring__nested_hole():
    nested_ring = [
        [4.0, 6.0],
        [6.0, 6.0],
        [6.0, 4.0],
        [4.0, 4.0],
    ]
    polygon = sg.Polygon(shell=outer_coords, holes=[inner])
    inner_polygon = sg.Polygon(shell=inner_coords)
    polygons = gpd.GeoDataFrame(geometry=[polygon, inner_polygon])
    polygons["cellsize"] = 1.0
    linestrings = gpd.GeoDataFrame(geometry=[sg.LineString(nested_ring)])

    new_polygons = tg.convert_linestring_rings(polygons, linestrings)
    assert isinstance(new_polygons, gpd.GeoDataFrame)
    assert np.allclose(new_polygons["cellsize"], 1.0)
    assert np.allclose(new_polygons.area, [84.0, 12.0, 4.0])


def test_segmentize_linestrings():
    gdf = gpd.GeoDataFrame()
    actual = tg.segmentize_linestrings(gdf)
    assert actual is gdf

    gdf = gpd.GeoDataFrame(
        geometry=[
            sg.LineString([[0.0, 0.0], [10.0, 0.0]]),
            sg.LineString([[0.0, 5.0], [10.0, 5.0]]),
            sg.LineString([[0.0, 10.0], [10.0, 10.0]]),
        ],
        data={"cellsize": [np.nan, 1.0, 0.5]},
    )
    actual = tg.segmentize_linestrings(gdf)
    _, index = shapely.get_coordinates(actual.geometry, return_index=True)
    _, counts = np.unique(index, return_counts=True)
    assert np.array_equal(counts, [2, 11, 21])
