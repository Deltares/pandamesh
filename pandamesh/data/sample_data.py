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
)
with pkg_resources.resource_stream("pandamesh.data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def provinces_nl():
    """The provinces (including water bodies) of the Netherlands."""
    fname = REGISTRY.fetch("provinces-nl.geojson")
    return gpd.read_file(fname)
