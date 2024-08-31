import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg

from pandamesh import gmsh_geometry as gg
from pandamesh.gmsh_mesher import gmsh_env


class TestGmshGeometry:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.outer_coords = np.array(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        )
        self.inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
        self.line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
        self.line = sg.LineString(self.line_coords)
        self.polygon = sg.Polygon(self.outer_coords)
        self.donut = sg.Polygon(self.outer_coords, holes=[self.inner_coords])
        self.refined = sg.Polygon(self.inner_coords)
        y = np.arange(0.5, 10.0, 0.5)
        x = np.full(y.size, 1.0)
        self.points_embed = gpd.points_from_xy(x, y)
        self.line_embed = sg.LineString(
            [
                [9.0, 2.0],
                [9.0, 8.0],
            ]
        )
        self.polygons = gpd.GeoDataFrame(
            {"cellsize": [1.0], "__polygon_id": [1]}, geometry=[self.donut]
        )

    def test_polygon_info(self):
        info, vertices, cellsizes, index = gg.polygon_info(self.polygon, 1.0, 0, 0)
        expected_info = gg.PolygonInfo(0, 4, [], [], 0)
        assert np.allclose(vertices, self.outer_coords)
        assert info == expected_info
        assert np.allclose(cellsizes, 1.0)
        assert index == 4

        info, vertices, cellsizes, index = gg.polygon_info(self.donut, 1.0, 0, 0)
        expected_info = gg.PolygonInfo(0, 4, [4], [4], 0)
        assert np.allclose(vertices, [self.outer_coords, self.inner_coords])
        assert info == expected_info
        assert np.allclose(cellsizes, 1.0)
        assert index == 8

    def test_linestring_info(self):
        info, vertices, cellsizes, index = gg.linestring_info(self.line, 1.0, 0, 0)
        expected_info = gg.LineStringInfo(0, 2, 0)
        assert np.allclose(vertices, self.line_coords)
        assert info == expected_info
        assert np.allclose(cellsizes, 1.0)
        assert index == 2

    def test_add_vertices(self):
        with gmsh_env():
            gg.add_vertices(
                [(0.0, 0.0), (1.0, 1.0)],
                [1.0, 1.0],
                [1, 2],
            )

    def test_add_linestrings(self):
        info = gg.LineStringInfo(0, 2, 0)
        with gmsh_env():
            gg.add_vertices(
                [(0.0, 0.0), (1.0, 1.0)],
                [1.0, 1.0],
                [0, 1],
            )
            gg.add_linestrings([info], [0, 1])

    def test_add_curve_loop(self):
        with gmsh_env():
            gg.add_vertices(
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                [1.0, 1.0, 1.0, 1.0],
                [0, 1, 2, 3],
            )
            curve_loop_tag = gg.add_curve_loop([0, 1, 2, 3])
        assert curve_loop_tag == 1

    def test_add_polygons(self):
        info = gg.PolygonInfo(0, 4, [4], [4], 0)
        vertices = np.vstack([self.inner_coords, self.outer_coords])
        cellsizes = np.full(vertices.size, 1.0)
        tags = np.arange(vertices.size)
        with gmsh_env():
            gg.add_vertices(vertices, cellsizes, tags)
            loop_tags, plane_tags = gg.add_polygons([info], tags)
        assert loop_tags == [1, 2]
        assert plane_tags == [0]

    def test_add_points(self):
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

    def test_collect_polygons(self):
        gdf = gpd.GeoDataFrame(geometry=[self.polygon])
        gdf["cellsize"] = 1.0
        gdf["__polygon_id"] = 1
        index, vertices, cellsizes, features = gg.collect_polygons(gdf, 0)
        assert index == 4
        assert np.allclose(vertices, self.outer_coords)
        assert np.allclose(cellsizes, 1.0)
        assert features == [gg.PolygonInfo(0, 4, [], [], 1)]

    def test_collect_linestrings(self):
        gdf = gpd.GeoDataFrame(geometry=[self.line])
        gdf["cellsize"] = 1.0
        gdf["__polygon_id"] = 1
        index, vertices, cellsizes, features = gg.collect_linestrings(gdf, 0)
        assert index == 2
        assert np.allclose(vertices, self.line_coords)
        assert np.allclose(cellsizes, 1.0)
        assert features == [gg.LineStringInfo(0, 2, 1)]

    def test_collect_points(self):
        x = np.arange(0.5, 10.0, 0.5)
        y = np.full(x.size, 1.0)
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
        xy = gg.collect_points(gdf)
        assert np.allclose(xy, np.column_stack([x, y]))

    def test_embed_where_linestring(self):
        line_gdf = gpd.GeoDataFrame({"cellsize": [0.5]}, geometry=[self.line_embed])
        actual = gg.embed_where(line_gdf, self.polygons)
        assert np.allclose(actual["cellsize"], 0.5)
        assert np.allclose(actual["__polygon_id"], 1)
        assert actual.geometry.iloc[0] == self.line_embed

    def test_embed_where_points(self):
        points_gdf = gpd.GeoDataFrame(geometry=self.points_embed)
        points_gdf["cellsize"] = 0.25
        actual = gg.embed_where(points_gdf, self.polygons)
        assert np.allclose(actual["cellsize"], 0.25)
        assert np.allclose(actual["__polygon_id"], 1)
        assert (actual.geometry.to_numpy() == self.points_embed).all()

    def test_add_geometry(self):
        line_gdf = gpd.GeoDataFrame({"cellsize": [0.5]}, geometry=[self.line_embed])
        points_gdf = gpd.GeoDataFrame(geometry=self.points_embed)
        points_gdf["cellsize"] = 0.25
        with gmsh_env():
            gg.add_geometry(self.polygons, line_gdf, points_gdf)

    def test_add_distance_points(self):
        with gmsh_env():
            indices = gg.add_distance_points(self.points_embed)
        assert np.array_equal(indices, np.arange(1, 20))

    def test_add_distance_linestring(self):
        with gmsh_env():
            indices = gg.add_distance_linestring(self.line, distance=20)
        assert np.array_equal(indices, [1])

        with gmsh_env():
            indices = gg.add_distance_linestring(self.line, distance=5)
        assert np.array_equal(indices, [1, 2])

        with gmsh_env():
            indices = gg.add_distance_linestring(self.line, distance=3)
        assert np.array_equal(indices, [1, 2, 3])

    def test_add_distance_linestrings(self):
        lines = gpd.GeoSeries([self.line, self.line_embed])
        with gmsh_env():
            indices = gg.add_distance_linestrings(lines, spacing=np.array([20, 20]))
        assert np.array_equal(indices, [1, 2])

        with gmsh_env():
            indices = gg.add_distance_linestrings(lines, spacing=np.array([3, 3]))
        assert np.array_equal(indices, [1, 2, 3, 4, 5])

        with gmsh_env():
            indices = gg.add_distance_linestrings(lines, spacing=np.array([20, 3]))
        assert np.array_equal(indices, [1, 2, 3])

    def test_add_distance_polygons(self):
        polygons = gpd.GeoSeries([self.donut, self.refined])
        with gmsh_env():
            indices = gg.add_distance_polygons(polygons, spacing=np.array([100, 100]))
        # Exterior, interior, exterior
        assert np.array_equal(indices, [1, 2, 3])

        with gmsh_env():
            indices = gg.add_distance_polygons(polygons, spacing=np.array([3, 3]))
        # Exterior, interior, exterior
        assert np.array_equal(indices, np.arange(1, 27))

    def test_add_distance_geometry(self):
        geometry = gpd.GeoSeries([sg.MultiPolygon([self.donut, self.refined])])
        with pytest.raises(TypeError, match="Geometry should be one of"):
            gg.add_distance_geometry(geometry, np.array([1.0]))

        # One point on each edge
        geometry = gpd.GeoSeries([self.polygon])
        with gmsh_env():
            indices = gg.add_distance_geometry(geometry, np.array([10.0]))
        assert np.array_equal(indices, [1, 2, 3, 4])

        # Three points
        geometry = gpd.GeoSeries([self.line])
        with gmsh_env():
            indices = gg.add_distance_geometry(geometry, np.array([3.0]))
        assert np.array_equal(indices, [1, 2, 3])

        # Twenty points, one to one
        geometry = gpd.GeoSeries(self.points_embed)
        with gmsh_env():
            indices = gg.add_distance_geometry(geometry, np.array([np.nan]))
        assert np.array_equal(indices, np.arange(1, 20))

        # All together
        geometry = gpd.GeoSeries(
            np.concatenate([[self.polygon], [self.line], self.points_embed])
        )
        with gmsh_env():
            indices = gg.add_distance_geometry(geometry, np.array([10.0, 3.0, np.nan]))
        assert np.array_equal(indices, np.arange(1, 25))
