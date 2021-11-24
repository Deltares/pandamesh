import textwrap
from enum import Enum

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg

from pandamesh import common


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


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
Le = sg.LineString(
    [
        (3.0, 0.0),
        (4.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]
)


pa = sg.Point(0.5, 0.5)
pb = sg.Point(0.5, 1.5)


def test_flatten():
    assert common.flatten([[]]) == []
    assert common.flatten([[1]]) == [1]
    assert common.flatten([[1], [2, 3]]) == [1, 2, 3]


def test_show_options():
    common._show_options(Color) == textwrap.dedent(
        """
        RED
        GREEN
        BLUE
    """
    )


def test_invalid_option():
    common.invalid_option("YELLOW", Color) == textwrap.dedent(
        """
        Invalid option: YELLOW. Valid options are:
        RED
        GREEN
        BLUE
    """
    )


def test_overlap_shortlist():
    # b overlaps with a
    polygons = gpd.GeoSeries(data=[a, b, d], index=[0, 1, 2])
    ia, ib = common.overlap_shortlist(polygons)
    assert np.array_equal(ia, [0])
    assert np.array_equal(ib, [1])

    # a touches c, c touches d: should count as overlap for shortlist.
    polygons = gpd.GeoSeries(data=[a, c, d], index=[0, 1, 2])
    ia, ib = common.overlap_shortlist(polygons)
    assert np.array_equal(ia, [0, 1])
    assert np.array_equal(ib, [1, 2])

    linestrings = gpd.GeoSeries(data=[La, Lb, Lc], index=[0, 1, 2])
    ia, ib = common.overlap_shortlist(linestrings)
    assert np.array_equal(ia, [0])
    assert np.array_equal(ib, [1])


def test_intersecting_features():
    polygons = gpd.GeoSeries(data=[a, b, c, d], index=[0, 1, 2, 3])
    ia, ib = common.intersecting_features(polygons, "polygon")
    assert np.array_equal(ia, [0, 1])
    assert np.array_equal(ib, [1, 2])

    linestrings = gpd.GeoSeries(data=[La, Lb, Lc], index=[0, 1, 2])
    ia, ib = common.intersecting_features(linestrings, "linestring")
    assert np.array_equal(ia, [0])
    assert np.array_equal(ib, [1])


def test_check_linestrings():
    polygons = gpd.GeoSeries(data=[a, c, d], index=[0, 1, 2])
    linestrings = gpd.GeoSeries(data=[La, Lb, Lc, Le], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="1 cases of complex linestring detected"):
        common.check_linestrings(linestrings, polygons)

    linestrings = gpd.GeoSeries(data=[La, Lb, Lc], index=[0, 1, 2])
    with pytest.raises(ValueError, match="1 cases of intersecting linestring detected"):
        common.check_linestrings(linestrings, polygons)

    linestrings = gpd.GeoSeries(data=[La, Ld], index=[0, 1])
    with pytest.raises(
        ValueError, match="The same linestring detected in multiple polygons"
    ):
        common.check_linestrings(linestrings, polygons)

    polygons = gpd.GeoSeries(data=[a, c, d], index=[0, 1, 2])
    linestrings = gpd.GeoSeries(data=[La, Lc], index=[0, 1])
    common.check_linestrings(linestrings, polygons)


def test_check_polygons():
    polygons = gpd.GeoSeries(data=[a, b, c, d, e], index=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="1 cases of complex polygon detected"):
        common.check_polygons(polygons)

    polygons = gpd.GeoSeries(data=[a, b, c, d], index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="2 cases of intersecting polygon detected"):
        common.check_polygons(polygons)


def test_check_points():
    points = gpd.GeoSeries(data=[pa, pb], index=[0, 1])
    polygons = gpd.GeoSeries(data=[a, b, c, d, e], index=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="1 points detected outside"):
        common.check_points(points, polygons)


def test_separate_shapely():
    gdf = gpd.GeoDataFrame(geometry=[a, c, d, La, Lc, pa])
    polygons, linestrings, points = common.separate_shapely(gdf)
    assert isinstance(polygons.geometry.iloc[0], sg.Polygon)
    assert isinstance(linestrings.geometry.iloc[0], sg.LineString)
    assert isinstance(points.geometry.iloc[0], sg.Point)
