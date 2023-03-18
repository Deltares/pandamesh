"""
Functions to load sample data.
"""
import geopandas as gpd
import pkg_resources
import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("pandamesh"),
    base_url="https://github.com/deltares/xugrid/raw/main/data/",
    version=None,
    version_dev="main",
    env="PANDAMESH_DATA_DIR",
    registry={
        "provinces.geojson": "ac6f8bb30aa021a6c79c003de794e55a3636aa66b600722fa1b0d6f60a45caaf",
    }
)

def provinces_nl():
    """The provinces (including water bodies) of the Netherlands."""
    fname = REGISTRY.fetch("provinces-nl.geojson")
    return gpd.read_file(fname)
