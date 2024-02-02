"""Functions to load sample data."""
import geopandas as gpd
import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("pandamesh"),
    base_url="https://github.com/deltares/pandamesh/blob/main/data/",
    version=None,
    version_dev="main",
    env="PANDAMESH_DATA_DIR",
    registry={
        "provinces-nl.geojson": "7539318974d1d78f35e4c2987287aa81f5ff505f444a2e0f340d804f57c0f8e3",
        "south-america.geojson": "337746351d15a83d5d41f1cecd30aa40b1698eb7587a4e412c511af89f82e49c",
    },
)


def provinces_nl():
    """Return the provinces (including water bodies) of the Netherlands as a GeoDataframe."""
    fname = REGISTRY.fetch("provinces-nl.geojson")
    return gpd.read_file(fname)


def south_america():
    fname = REGISTRY.fetch("south-america.geojson")
    return gpd.read_file(fname)
