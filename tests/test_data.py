import geopandas as gpd

import pandamesh as pm


def test_provinces_nl():
    gdf = pm.data.provinces_nl()
    assert isinstance(gdf, gpd.GeoDataFrame)
