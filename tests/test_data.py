import geopandas as gpd

import pandamesh as pm


def test_provinces_nl():
    gdf = pm.data.provinces_nl()
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_south_america():
    gdf = pm.data.south_america()
    assert isinstance(gdf, gpd.GeoDataFrame)
