"""
Functions to load sample data.
"""
import geopandas as gpd
import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("pandamesh"),
    base_url="https://github.com/deltares/pandamesh/raw/main/data/",
    version=None,
    version_dev="main",
    env="PANDAMESH_DATA_DIR",
    registry={
        "provinces-nl.geojson": "7539318974d1d78f35e4c2987287aa81f5ff505f444a2e0f340d804f57c0f8e3",
    },
)


def provinces_nl():
    """The provinces (including water bodies) of the Netherlands."""
    fname = REGISTRY.fetch("provinces-nl.geojson")
    return gpd.read_file(fname)
