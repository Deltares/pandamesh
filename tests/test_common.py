import re

import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.geometry as sg

from pandamesh import common

a = sg.Polygon(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
)
b = sg.Polygon(
    [
        (0.5, 0.0),
        (1.5, 0.0),
        (1.5, 1.0),
        (0.5, 1.0),
    ]
)
c = sg.Polygon(
    [
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
    ]
)
d = sg.Polygon(
    [
        (2.0, 0.0),
        (3.0, 0.0),
        (3.0, 1.0),
        (2.0, 1.0),
    ]
)
# Bowtie:
e = sg.Polygon(
    [
        (3.0, 0.0),
        (4.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]
)
# Almost dangling edge
f = sg.Polygon(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (1.0, 0.001),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
)

La = sg.LineString(
    [
        (0.25, 0.25),
        (0.75, 0.75),
    ]
)
Lb = sg.LineString(
    [
        (0.25, 0.75),
        (0.75, 0.25),
    ]
)
Lc = sg.LineString(
    [
        (2.25, 0.25),
        (2.75, 0.75),
    ]
)
Ld = sg.LineString(
    [
        (0.6, 0.5),
        (1.5, 0.5),
    ]
)
# Bowtie
Le = sg.LineString(
    [
        (3.0, 0.0),
        (4.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]
)
# Outside
Lf = sg.LineString(
    [
        (1.0, 2.0),
        (2.0, 2.0),
    ]
)

Ra = sg.LinearRing(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
)


pa = sg.Point(0.5, 0.5)
pb = sg.Point(0.5, 1.5)


def test_flatten():
    assert common.flatten([[]]) == []
    assert common.flatten([[1]]) == [1]
    assert common.flatten([[1], [2, 3]]) == [1, 2, 3]


def test_flatten_geometry():
    point = sg.Point(0, 0)
    assert common.flatten_geometry(point) == [point]

    multi_point = sg.MultiPoint([sg.Point(0, 0), sg.Point(1, 1)])
    assert common.flatten_geometry(multi_point) == list(multi_point.geoms)

    point = sg.Point(0, 0)
    line = sg.LineString([(1, 1), (2, 2)])
    polygon = sg.Polygon([(3, 3), (4, 4), (4, 3)])
    multi_point = sg.MultiPoint([sg.Point(5, 5), sg.Point(6, 6)])

    collection = sg.GeometryCollection([point, line, polygon, multi_point])
    expected = [point, line, polygon, sg.Point(5, 5), sg.Point(6, 6)]
    assert common.flatten_geometry(collection) == expected

    inner_collection = sg.GeometryCollection([point, multi_point])
    nested_collection = sg.GeometryCollection([inner_collection, line, polygon])
    expected = [point, sg.Point(5, 5), sg.Point(6, 6), line, polygon]
    assert common.flatten_geometry(nested_collection) == expected

    # Also test flatten_geometries
    geometries = [
        point,
        sg.GeometryCollection([multi_point]),
        sg.GeometryCollection([line, polygon]),
    ]
    actual = common.flatten_geometries(geometries)
    assert isinstance(actual, np.ndarray)
    assert all(actual == expected)


def test_check_geodataframe():
    with pytest.raises(TypeError, match="Expected GeoDataFrame"):
        common.check_geodataframe([1, 2, 3], {})

    gdf = gpd.GeoDataFrame(geometry=[pa, pb])
    with pytest.raises(
        ValueError,
        match=re.escape("These column(s) are required but are missing: cellsize"),
    ):
        common.check_geodataframe(gdf, {"cellsize"})

    gdf["cellsize"] = 1.0
    gdf.index = [0, "1"]
    with pytest.raises(ValueError, match="geodataframe index is not integer typed"):
        common.check_geodataframe(gdf, {"cellsize"}, check_index=True)

    empty = gdf.loc[[]]
    with pytest.raises(ValueError, match="Dataframe is empty"):
        common.check_geodataframe(empty, {"cellsize"})

    gdf.index = [0, 0]
    with pytest.raises(ValueError, match="geodataframe index contains duplicates"):
        common.check_geodataframe(gdf, {"cellsize"}, check_index=True)

    gdf.index = [0, 1]
    common.check_geodataframe(gdf, {"cellsize"}, check_index=True)


def test_intersecting_features():
    polygons = gpd.GeoSeries(data=[a, b, c, d], index=[0, 1, 2, 3])
    ia, ib = common.intersecting_features(polygons, "polygon")
    assert np.array_equal(ia, [0, 1])
    assert np.array_equal(ib, [1, 2])

    linestrings = gpd.GeoSeries(data=[La, Lb, Lc], index=[0, 1, 2])
    ia, ib = common.intersecting_features(linestrings, "linestring")
    assert np.array_equal(ia, [0])
    assert np.array_equal(ib, [1])


def test_check_lines():
    # Complex (self-intersecting) linestring
    linestrings = gpd.GeoSeries(data=[La, Lb, Lc, Le], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="1 cases of complex lines detected"):
        common.check_lines(linestrings)

    # Valid input
    linestrings = gpd.GeoSeries(data=[La, Lc], index=[0, 1])
    common.check_lines(linestrings)


def test_compute_intersections():
    segments = shapely.linestrings(
        [
            [
                [0.0, 5.0],
                [10.0, 5.0],
            ],
            [
                [5.0, 0.0],
                [5.0, 10.0],
            ],
            [
                [0.0, -5.0],
                [10.0, -5.0],
            ],
        ]
    )
    i = np.array([0, 0, 1, 1, 2])
    j = np.array([0, 1, 1, 2, 2])

    _i, _j, actual = common.compute_intersections(segments, i, j)
    assert np.array_equal(_i, [0])
    assert np.array_equal(_j, [1])
    assert np.array_equal(actual, [[5.0, 5.0]])

    # No intersections
    _i, _j, actual = common.compute_intersections(segments, i, i)
    assert np.array_equal(_i, [])
    assert np.array_equal(_j, [])
    assert np.array_equal(actual, np.array([]).reshape((-1, 2)))

    i = np.array([0])
    j = np.array([2])
    _i, _j, actual = common.compute_intersections(segments, i, j)
    assert np.array_equal(_i, [])
    assert np.array_equal(_j, [])
    assert np.array_equal(actual, np.array([]).reshape((-1, 2)))

    # Parallel
    segments = shapely.linestrings(
        [
            [
                [0.0, 5.0],
                [10.0, 5.0],
            ],
            [
                [3.0, 5.0],
                [13.0, 5.0],
            ],
        ]
    )
    i = np.array([0])
    j = np.array([1])
    _i, _j, actual = common.compute_intersections(segments, i, j)
    assert np.array_equal(_i, [0, 0])
    assert np.array_equal(_j, [1, 1])
    expected = np.array(
        [
            [3.0, 5.0],
            [10.0, 5.0],
        ]
    )
    assert np.array_equal(actual, expected)


def test_check_polygons():
    polygons = gpd.GeoSeries(data=[a, b, c, d, e], index=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="1 cases of complex polygon detected"):
        common.check_polygons(polygons, 1.0e-3)

    polygons = gpd.GeoSeries(data=[a, b, c, d], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="2 cases of intersecting polygon detected"):
        common.check_polygons(polygons, 1.0e-3)

    polygons = gpd.GeoSeries(data=[f], index=[0])
    with pytest.raises(
        ValueError, match="1 proximate points found on polygon perimeters"
    ):
        common.check_polygons(polygons, 1.0e-3)


def test_check_points():
    points = gpd.GeoSeries(data=[pa, pb], index=[0, 1])
    polygons = gpd.GeoSeries(data=[a, b, c, d, e], index=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="1 points detected outside"):
        common.check_points(points, polygons)


def test_separate_geometry():
    bad = np.array([sg.MultiPolygon([a, d])])
    with pytest.raises(
        TypeError, match="GeoDataFrame contains unsupported geometry types"
    ):
        common.separate_geometry(bad)

    geometry = np.array([a, c, La, Lc, pa, Ra, d])
    polygons, lines, points = common.separate_geometry(geometry)
    assert all(polygons == [a, c, d])
    assert all(lines == [La, Lc, Ra])
    assert all(points == [pa])


def test_separate_geodataframe():
    gdf = gpd.GeoDataFrame(geometry=[a, c, d, La, Lc, pa])
    gdf["cellsize"] = 1.0

    with pytest.raises(ValueError, match="intersecting_edges should be one of"):
        common.separate_geodataframe(
            gdf, intersecting_edges="abc", minimum_spacing=1.0e-3
        )

    polygons, linestrings, points = common.separate_geodataframe(
        gdf, intersecting_edges="error", minimum_spacing=1.0e-3
    )
    assert isinstance(polygons.geometry.iloc[0], sg.Polygon)
    assert isinstance(linestrings.geometry.iloc[0], sg.LineString)
    assert isinstance(points.geometry.iloc[0], sg.Point)

    # Make sure it works for single elements
    gdf = gpd.GeoDataFrame(geometry=[a])
    gdf["cellsize"] = 1.0
    common.separate_geodataframe(
        gdf, intersecting_edges="error", minimum_spacing=1.0e-3
    )

    gdf = gpd.GeoDataFrame(geometry=[a, La])
    gdf["cellsize"] = 1.0
    polygons, linestrings, points = common.separate_geodataframe(
        gdf, intersecting_edges="error", minimum_spacing=1.0e-3
    )

    # Make sure cellsize is cast to float
    gdf = gpd.GeoDataFrame(geometry=[a, La])
    gdf["cellsize"] = "1"
    dfs = common.separate_geodataframe(
        gdf, intersecting_edges="error", minimum_spacing=1.0e-3
    )
    for df in dfs:
        assert np.issubdtype(df["cellsize"].dtype, np.floating)

    with pytest.raises(
        TypeError, match="GeoDataFrame contains unsupported geometry types"
    ):
        gdf = gpd.GeoDataFrame(geometry=[sg.MultiPolygon([a, b])])
        common.separate_geodataframe(
            gdf, intersecting_edges="error", minimum_spacing=1.0e-3
        )


def test_central_origin():
    gdf = gpd.GeoDataFrame(geometry=[d])
    back, x, y = common.central_origin(gdf, False)
    assert gdf is not back
    assert x == 0
    assert y == 0

    back, x, y = common.central_origin(gdf, True)
    assert np.allclose(x, 2.5)
    assert np.allclose(y, 0.5)
    assert np.array_equal(back.total_bounds, [-0.5, -0.5, 0.5, 0.5])


def test_grouper():
    a = np.array([1, 2, 3, 4])
    a_index = np.array([0, 0, 1, 1])
    b = np.array([10, 20, 30, 40, 50])
    b_index = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="All arrays must be of the same length"):
        grouper = common.Grouper(a, [0, 0, 1], b, b_index)

    grouper = common.Grouper(a, a_index, b, b_index)

    # First group
    a_val, b_vals = next(grouper)
    assert a_val == 1
    assert np.array_equal(b_vals, np.array([10, 20]))

    # Second group
    a_val, b_vals = next(grouper)
    assert a_val == 2
    assert np.array_equal(b_vals, np.array([30, 40]))

    # As iterator
    grouper = common.Grouper(a, a_index, b, b_index)
    results = list(grouper)
    assert results[0][0] == 1
    assert np.array_equal(results[0][1], np.array([10, 20]))
    assert results[1][0] == 2
    assert np.array_equal(results[1][1], np.array([30, 40]))


def test_to_geodataframe():
    # Two triangles
    vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])

    gdf = common.to_geodataframe(vertices, faces)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf == 2)

    # Check the coordinates of the first triangle
    assert np.allclose(
        np.array(gdf.geometry[0].exterior.coords),
        np.array([(0, 0), (1, 0), (0, 1), (0, 0)]),
    )

    # Check the coordinates of the second triangle
    assert np.allclose(
        np.array(gdf.geometry[1].exterior.coords),
        np.array([(1, 0), (1, 1), (0, 1), (1, 0)]),
    )

    # Define vertices
    vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0.5]], dtype=float)

    # Define faces (triangle: 0,1,2 and quadrangle: 1,3,4,2)
    faces = np.array([[0, 1, 2, -1], [1, 3, 4, 2]], dtype=int)

    # Call the function
    gdf = common.to_geodataframe(vertices, faces)

    # Assertions
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2  # Two polygons (one triangle, one quadrangle)


class TestLineworkIntersection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.donut = sg.Polygon(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            holes=[
                [
                    [2.0, 5.0],
                    [5.0, 8.0],
                    [8.0, 5.0],
                    [5.0, 2.0],
                ]
            ],
        )
        self.line0 = shapely.LineString(
            [
                [-2.0, 0.0],
                [12.0, 10.0],
            ]
        )
        self.line1 = shapely.LineString(
            [
                [5.5, 9.0],
                [9.0, 5.5],
            ]
        )
        self.gdf = gpd.GeoDataFrame(geometry=[self.donut, self.line0, self.line1])

    def test_find_edge_intersections(self):
        gdf = gpd.GeoDataFrame(geometry=[self.donut])
        with pytest.raises(TypeError, match="Expected geopandas.GeoSeries"):
            common.find_edge_intersections(gdf)

        actual = common.find_edge_intersections(gdf.geometry)
        assert len(actual) == 0

        gdf = gpd.GeoDataFrame(geometry=[self.donut, self.line0])
        actual = common.find_edge_intersections(gdf.geometry)
        assert len(actual) == 4

        gdf = gpd.GeoDataFrame(geometry=[self.donut, self.line1])
        actual = common.find_edge_intersections(gdf.geometry)
        assert len(actual) == 0

    def test_check_linework(self):
        gdf = gpd.GeoDataFrame(geometry=[self.donut, self.line1])
        polygons, lines, _ = common.separate_geometry(gdf.geometry)
        common.check_linework(polygons, lines, "error")

        gdf = gpd.GeoDataFrame(geometry=[self.donut, self.line0])
        polygons, lines, _ = common.separate_geometry(gdf.geometry)
        with pytest.raises(ValueError):
            common.check_linework(polygons, lines, "error")

        with pytest.warns(UserWarning):
            common.check_linework(polygons, lines, "warn")


class TestProximatePoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shell0 = [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [15.0, 0.0],
            [10.0, 0.1],
            [10.0, 9.9],
            [13.0, 10.0],
            [10.0, 10.0],
            [5.0, 10.0],
            [0.0, 10.0],
        ]
        self.shell1 = [
            [20.0, 0.0],
            [30.0, 0.0],
            [30.0, 5.0],
            [25.0, 5.0],
            [30.0, 5.1],
            [30.0, 10.0],
            [20.0, 10.0],
        ]
        self.shell2 = [
            [40.0, 0.0],
            [50.0, 0.0],
            [50.0, 10.0],
            [40.0, 10.0],
        ]
        self.hole2 = [
            [42.0, 2.0],
            [42.0, 8.0],
            [45.0, 8.0],
            [45.0, 7.0],
            [45.1, 8.0],
            [48.0, 8.0],
            [48.0, 2.0],
        ]
        self.shell3 = [
            [60.0, 0.0],
            [70.0, 0.0],
            [73.0, 0.0],
            [73.0, 0.01],
            [70.0, 0.01],
            [70.0, 10.0],
            [60.0, 10.0],
        ]

    # Order of vertices shouldn't matter, since we're calling shapely.normalize
    @pytest.mark.parametrize("flip", [True, False])
    def test_find_proximate_points(self, flip: bool):
        def construct_polygon(shell, hole=None):
            if flip:
                shell = reversed(shell)
                if hole is not None:
                    hole = reversed(hole)
            return sg.Polygon(shell=shell, holes=[hole])

        poly0 = construct_polygon(self.shell0)
        poly1 = construct_polygon(self.shell1)
        poly2 = construct_polygon(self.shell2, self.hole2)
        poly3 = construct_polygon(self.shell3)
        geometry = gpd.GeoSeries([poly0, poly1, poly2, poly3])

        # Default tolerance of 0.001, no problems
        faulty = common.find_proximate_perimeter_points(geometry)
        assert isinstance(faulty, gpd.GeoSeries)
        assert len(faulty) == 0

        faulty = common.find_proximate_perimeter_points(geometry, 0.5)
        expected = np.array(
            [
                [13.0, 10.0],
                [15.0, 0.0],
                [73.0, 0.01],
                [73.0, 0.0],
                [45.0, 7.0],
            ]
        )
        assert isinstance(faulty, gpd.GeoSeries)
        assert len(faulty) == 5
        actual = shapely.get_coordinates(faulty)
        assert np.array_equal(actual, expected)
