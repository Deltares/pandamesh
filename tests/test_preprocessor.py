import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.geometry as sg

from pandamesh import preprocessor as pr

outer = sg.Polygon(
    [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ]
)
inner = sg.Polygon(
    [
        [5.0, 2.0],
        [8.0, 5.0],
        [5.0, 8.0],
        [2.0, 5.0],
    ]
)

first = sg.Polygon(
    [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ]
)
second = sg.Polygon(
    [
        [10.0, 2.0],
        [18.0, 2.0],
        [18.0, 8.0],
        [10.0, 8.0],
    ]
)
third = sg.Polygon(
    [
        [18.0, 2.0],
        [22.0, 2.0],
        [22.0, 8.0],
        [18.0, 8.0],
    ]
)

donut = sg.Polygon(
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

first_line = shapely.LineString(
    [
        [-5.0, 5.0],
        [6.0, 5.0],
    ]
)
second_line = shapely.LineString(
    [
        [5.0, -5.0],
        [5.0, 6.0],
    ]
)


def test_collect_exteriors():
    polygons = np.array([outer, donut])
    exteriors = pr.collect_exteriors(polygons)
    assert len(exteriors) == 2
    assert (shapely.get_type_id(exteriors) == 2).all()


def test_collect_interiors():
    polygons = np.array([donut, outer])
    interiors = pr.collect_interiors(polygons)
    assert len(interiors == 1)
    assert (shapely.get_type_id(interiors) == 2).all()


def test_filter_holes_and_assign_values():
    # Assume: two polygons. Union, then polygonized to create a partitions. Any
    # polygon found in a hole should disappear.
    polygons = np.array([donut, inner, second, third])
    # Overlap between the original outer and inner polygon.
    index_original = np.array([0, 1, 0, 1])
    index_union = np.array([0, 0, 1, 1])
    indexer = np.array([4, 3, 2, 1])

    filtered_polygons, filtered_indexer = pr.filter_holes_and_assign_values(
        polygons, index_original, index_union, indexer, ascending=True
    )
    assert all(filtered_polygons == [donut, inner])
    assert np.array_equal(filtered_indexer, [3, 3])

    filtered_polygons, filtered_indexer = pr.filter_holes_and_assign_values(
        polygons, index_original, index_union, indexer, ascending=False
    )
    assert all(filtered_polygons == [donut, inner])
    assert np.array_equal(filtered_indexer, [4, 4])


def test_locate_polygons():
    new = np.array([donut, inner])
    old = np.array([outer, inner])
    indexer = np.array([1, 0])
    polygons, indexer = pr.locate_polygons(new, old, indexer, True)
    assert all(polygons == [donut, inner])
    assert np.array_equal(indexer, [1, 0])

    polygons, indexer = pr.locate_polygons(new, old, indexer, False)
    assert all(polygons == [donut, inner])
    assert np.array_equal(indexer, [1, 1])


def test_locate_lines():
    old = np.array([first_line, second_line])
    new = np.array(
        [
            shapely.LineString([[-5.0, 5.0], [0.0, 5.0]]),
            shapely.LineString([[5.0, -5.0], [5.0, 0.0]]),
            shapely.LineString([[5.0, 0.0], [5.0, 5.0]]),
            shapely.LineString([[0.0, 5.0], [5.0, 5.0]]),
        ]
    )
    indexer = np.array([0, 1])
    located = pr.locate_lines(new, old, indexer)
    assert np.array_equal(located, [0, 1, 1, 0])


def test_merge_polyons():
    polygons = np.array([first, second])
    merged = pr.merge_polygons(polygons, None)
    assert len(merged == 1)
    assert (shapely.get_type_id(merged) == 3).all()

    polygons = np.array([inner, second])
    merged = pr.merge_polygons(polygons, None)
    assert len(merged == 2)
    assert (shapely.get_type_id(merged) == 3).all()


def test_preprocessor_init():
    geometry = [outer]
    values = [1.0, 2.0]

    with pytest.raises(ValueError, match="geometry and values shape mismatch"):
        pr.Preprocessor(geometry, values)

    point = sg.Point([1.0, 1.0])
    geometry = [outer, inner, second, first_line, point]
    p = pr.Preprocessor(geometry)
    assert p.values is None
    assert isinstance(p.polygons, np.ndarray)
    assert isinstance(p.lines, np.ndarray)
    assert isinstance(p.points, np.ndarray)
    assert np.array_equal(p.polygon_indexer, [0, 1, 2])
    assert np.array_equal(p.line_indexer, [3])
    assert np.array_equal(p.point_indexer, [4])

    values = [3.0, 2.0, 1.0, 0.5, 0.5]
    p = pr.Preprocessor(geometry, values)
    assert np.array_equal(p.values, [0.5, 1.0, 2.0, 3.0])
    assert np.array_equal(p.polygon_indexer, [3, 2, 1])
    assert np.array_equal(p.line_indexer, [0])
    assert np.array_equal(p.point_indexer, [0])


class TestPreprocessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        points = [sg.Point([0.1, 0.1]), sg.Point([1.0, 1.0]), sg.Point([-5.0, 5.0])]
        self.geometry = [outer, inner, second, first_line, second_line, *points]
        self.values = [3.0, 1.0, 3.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        self.p = pr.Preprocessor(self.geometry)
        self.vp = pr.Preprocessor(self.geometry, self.values)

    def test_copy_with(self):
        def test_value_equality(old, new):
            for k, v in new.items():
                assert v is old[k]

        p = self.p
        copied = p._copy_with()
        assert isinstance(copied, pr.Preprocessor)
        assert copied is not p
        test_value_equality(p.__dict__, copied.__dict__)

        copied = p._copy_with(values=[1, 2, 3, 4, 5, 6, 7])
        assert p.values is not copied.values
        new = copied.__dict__
        assert np.array_equal(new.pop("values"), [1, 2, 3, 4, 5, 6, 7])
        test_value_equality(p.__dict__, new)

    def test_to_geodataframe(self):
        gdf = self.p.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert set(gdf.columns) == {"geometry", "indexer"}
        assert np.array_equal(gdf["indexer"], [0, 1, 2, 3, 4, 5, 6, 7])

        gdf = self.vp.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert set(gdf.columns) == {"geometry", "indexer", "values"}
        assert np.array_equal(gdf["values"], self.values)

    def test_merge_polygons(self):
        merged = self.p.merge_polygons()
        assert isinstance(merged, pr.Preprocessor)
        assert len(merged.polygons == self.p.polygons)
        assert (shapely.get_type_id(merged.polygons) == 3).all()

        merged = self.vp.merge_polygons()
        assert len(merged.polygons) == 2
        assert np.array_equal(merged.polygon_indexer, [1, 2])
        assert (shapely.get_type_id(merged.polygons) == 3).all()

    def test_unify_polygons(self):
        unified = self.p.unify_polygons()
        assert isinstance(unified, pr.Preprocessor)
        assert len(unified.polygons) == 5
        assert (shapely.get_type_id(unified.polygons) == 3).all()
        assert np.array_equal(unified.polygon_indexer, [0, 0, 2, 0, 0])

        # Outer and second have same values, so they are merged.
        unified = self.vp.unify_polygons().merge_polygons()
        assert len(unified.polygons) == 2
        assert (shapely.get_type_id(unified.polygons) == 3).all()
        assert np.array_equal(unified.polygon_indexer, [1, 2])

    def test_clip_lines(self):
        clipped = self.p.clip_lines()
        assert isinstance(clipped, pr.Preprocessor)
        assert len(clipped.lines) == 2

        expected = [
            sg.LineString([[0.0, 5.0], [6.0, 5.0]]),
            sg.LineString([[5.0, 0.0], [5.0, 6.0]]),
        ]
        assert all(clipped.lines == expected)
        assert np.array_equal(clipped.line_indexer, [3, 4])

        p = pr.Preprocessor(geometry=[donut, first_line, second_line])
        clipped = p.clip_lines(distance=0.5)
        expected = [
            sg.LineString([[0.5, 5.0], [1.5, 5.0]]),
            sg.LineString([[5.0, 0.5], [5.0, 1.5]]),
        ]
        assert all(clipped.lines == expected)
        assert np.array_equal(clipped.line_indexer, [1, 2])

    def test_unify_lines(self):
        unified = self.p.unify_lines()
        assert isinstance(unified, pr.Preprocessor)
        assert len(unified.lines) == 4
        expected = [
            sg.LineString([[-5.0, 5.0], [5.0, 5.0]]),
            sg.LineString([[5.0, 5.0], [6.0, 5.0]]),
            sg.LineString([[5.0, -5.0], [5.0, 5.0]]),
            sg.LineString([[5.0, 5.0], [5.0, 6.0]]),
        ]
        assert all(unified.lines == expected)
        assert np.array_equal(unified.line_indexer, [3, 3, 4, 4])

    def test_clip_points(self):
        clipped = self.p.clip_points()
        assert isinstance(clipped, pr.Preprocessor)
        assert len(clipped.points) == 2
        assert np.array_equal(clipped.point_indexer, [5, 6])

        clipped = self.p.clip_points(distance=0.5)
        assert isinstance(clipped, pr.Preprocessor)
        assert len(clipped.points) == 1
        assert np.array_equal(clipped.point_indexer, [6])

    def test_interpolate_lines_to_points(self):
        with pytest.raises(ValueError, match="If values_as_distance is False"):
            self.p.interpolate_lines_to_points(distance=None, values_as_distance=False)

        interpolated = self.p.interpolate_lines_to_points(distance=1.0)
        assert isinstance(interpolated, pr.Preprocessor)
        assert len(interpolated.points) == 27
        assert np.array_equal(
            interpolated.point_indexer, [5, 6, 7] + [3] * 12 + [4] * 12
        )

        interpolated = self.vp.interpolate_lines_to_points(values_as_distance=True)
        assert isinstance(interpolated, pr.Preprocessor)
        assert len(interpolated.points) == 49
        assert np.array_equal(
            interpolated.point_indexer, [1, 1, 1] + [0] * 23 + [0] * 23
        )

    def test_snap_points(self):
        snapped = self.p.snap_points(0.05)
        assert isinstance(snapped, pr.Preprocessor)
        assert len(snapped.points) == 3
        assert np.array_equal(snapped.point_indexer, [5, 6, 7])

        snapped = self.p.snap_points(2.0)
        assert isinstance(snapped, pr.Preprocessor)
        assert len(snapped.points) == 2
        assert np.array_equal(snapped.point_indexer, [5, 7])

    def test_empty_ops(self):
        p = pr.Preprocessor(geometry=[outer])
        assert isinstance(p.clip_points(), pr.Preprocessor)
        assert isinstance(p.snap_points(1.0), pr.Preprocessor)
        assert isinstance(p.clip_lines(), pr.Preprocessor)
        assert isinstance(p.unify_lines(), pr.Preprocessor)
        assert isinstance(p.interpolate_lines_to_points(), pr.Preprocessor)
        assert isinstance(p.unify_polygons(), pr.Preprocessor)
        assert isinstance(p.merge_polygons(), pr.Preprocessor)
        assert isinstance(p.to_geodataframe(), gpd.GeoDataFrame)
