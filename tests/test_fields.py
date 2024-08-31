import re
import struct
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg
import xarray as xr

from pandamesh import gmsh_fields as gf
from pandamesh.gmsh_mesher import GmshMesher


@pytest.fixture(scope="function")
def gdf():
    polygon = sg.Polygon(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
    )
    return gpd.GeoDataFrame(geometry=[polygon], data={"cellsize": [10.0]})


def assert_refinement(mesh: gpd.GeoDataFrame):
    assert (mesh.area.max() / mesh.area.min()) > 5


class StructuredFieldContent(NamedTuple):
    cellsize: np.ndarray
    xmin: float
    ymin: float
    dx: float
    dy: float


def read_structured_field_file(
    path: Path
) -> tuple[np.ndarray, float, float, float, float]:
    with path.open("rb") as f:
        xmin, ymin, _ = struct.unpack("3d", f.read(24))  # 3 doubles, 8 bytes each
        dx, dy, _ = struct.unpack("3d", f.read(24))  # 3 doubles, 8 bytes each
        nrow, ncol, _ = struct.unpack("3i", f.read(12))  # 3 integers, 4 bytes each
        cellsize_data = f.read()  # Read the rest of the file

    cellsize = np.frombuffer(cellsize_data, dtype=np.float64).reshape((ncol, nrow))
    return StructuredFieldContent(cellsize, xmin, ymin, dx, dy)


def test_write_structured_field_file(tmp_path):
    cellsize = np.arange(6.0).reshape(2, 3)
    xmin = 5.0
    ymin = 15.0
    path = tmp_path / "a.dat"
    with pytest.raises(ValueError, match=re.escape("`cellsize` must be 2D")):
        gf.write_structured_field_file(path, cellsize[0], xmin, ymin, 1.0, 1.0)

    gf.write_structured_field_file(path, cellsize, xmin, ymin, 1.0, 1.0)
    assert path.exists()
    back = read_structured_field_file(path)
    assert back.dx == 1.0
    assert back.dy == 1.0
    assert back.xmin == 5.0
    assert back.ymin == 15.0
    # Gmsh expects it in column major order
    assert np.array_equal(back.cellsize, cellsize.T)

    path = tmp_path / "b.dat"
    gf.write_structured_field_file(path, cellsize, xmin, ymin, 1.0, -1.0)
    back = read_structured_field_file(path)
    assert back.dy == 1.0
    assert np.array_equal(back.cellsize[0, :], [3, 0])
    assert np.array_equal(back.cellsize[1, :], [4, 1])
    assert np.array_equal(back.cellsize[2, :], [5, 2])

    path = tmp_path / "c.dat"
    gf.write_structured_field_file(path, cellsize, xmin, ymin, -1.0, 1.0)
    back = read_structured_field_file(path)
    assert back.dx == 1.0
    assert np.array_equal(back.cellsize[0, :], [2, 5])
    assert np.array_equal(back.cellsize[1, :], [1, 4])
    assert np.array_equal(back.cellsize[2, :], [0, 3])


def test_math_eval_field(gdf):
    mesher = GmshMesher._force_init(gdf)

    with pytest.raises(TypeError):
        mesher.add_matheval_distance_field(1)

    field = gpd.GeoDataFrame(
        geometry=[sg.Point([5.0, 5.0])],
    )
    # missing columns
    with pytest.raises(ValueError):
        mesher.add_matheval_distance_field(field)

    field["spacing"] = np.nan
    field["function"] = "dist + 0.1"

    with pytest.raises(ValueError, match="distance not in MathEval field function"):
        mesher.add_matheval_distance_field(field)

    field["function"] = "max(distance, 0.5)"
    mesher.add_matheval_distance_field(field)
    mesh = mesher.generate_geodataframe()

    field = mesher.fields[0]
    assert isinstance(field, gf.MathEvalField)

    # Test whether anything has been refined.
    assert_refinement(mesh)

    # Assert combination field has been generated
    assert mesher._combination_field is not None
    assert isinstance(mesher._combination_field, gf.CombinationField)

    mesher.clear_fields()
    assert len(mesher.fields) == 0
    assert mesher._combination_field is None


def test_threshold_field(gdf):
    mesher = GmshMesher._force_init(gdf)

    with pytest.raises(TypeError):
        mesher.add_threshold_distance_field(1)

    field = gpd.GeoDataFrame(
        geometry=[sg.Point([5.0, 5.0])],
    )
    # missing columns
    with pytest.raises(ValueError):
        mesher.add_threshold_distance_field(field)

    field["dist_min"] = 1.0
    field["dist_max"] = 2.5
    field["size_min"] = 0.2
    field["size_max"] = 2.5
    field["spacing"] = np.nan

    mesher.add_threshold_distance_field(field)
    mesh = mesher.generate_geodataframe()

    # Test whether anything has been refined.
    assert_refinement(mesh)

    # Should work WITH optional columns as well
    mesher.clear_fields()
    field["sigmoid"] = True
    field["stop_at_dist_max"] = True

    mesher.add_threshold_distance_field(field)

    field = mesher.fields[0]
    assert isinstance(field, gf.ThresholdField)

    mesh = mesher.generate_geodataframe()
    assert isinstance(mesh, gpd.GeoDataFrame)


def test_line_field(gdf):
    line = sg.LineString(
        [
            [3.0, -3.0],
            [3.0, 13.0],
        ]
    )
    field = gpd.GeoDataFrame(geometry=[line])
    field["dist_min"] = 2.0
    field["dist_max"] = 4.0
    field["size_min"] = 0.5
    field["size_max"] = 2.5
    field["spacing"] = 2.0

    mesher = GmshMesher._force_init(gdf)
    mesher.add_threshold_distance_field(field)

    field = mesher.fields[0]
    assert isinstance(field, gf.ThresholdField)

    mesh = mesher.generate_geodataframe()
    assert_refinement(mesh)


def test_polygon_field(gdf):
    square = sg.Polygon(
        [
            [3.0, 3.0],
            [7.0, 3.0],
            [7.0, 7.0],
            [3.0, 7.0],
        ]
    )
    field = gpd.GeoDataFrame(geometry=[square])
    field["dist_min"] = 0.5
    field["dist_max"] = 1.5
    field["size_min"] = 0.3
    field["size_max"] = 2.5
    field["spacing"] = 1.0

    mesher = GmshMesher._force_init(gdf)
    mesher.add_threshold_distance_field(field)

    field = mesher.fields[0]
    assert isinstance(field, gf.ThresholdField)

    mesh = mesher.generate_geodataframe()
    assert_refinement(mesh)


def test_add_structured_field(gdf):
    mesher = GmshMesher._force_init(gdf)

    y, x = np.meshgrid([1.0, 5.0, 9.0], [1.0, 5.0, 9.0], indexing="ij")
    distance_from_origin = np.sqrt((x * x + y * y))
    cellsize = np.log(distance_from_origin / distance_from_origin.min()) + 0.5

    with pytest.raises(ValueError, match=r"Minimum cellsize must be > 0, received:"):
        mesher.add_structured_field(
            cellsize=cellsize * -1,
            xmin=x.min(),
            ymin=y.min(),
            dx=1.0,
            dy=1.0,
        )

    mesher.add_structured_field(
        cellsize=cellsize,
        xmin=x.min(),
        ymin=y.min(),
        dx=1.0,
        dy=1.0,
    )

    field = mesher.fields[0]
    assert isinstance(field, gf.StructuredField)
    assert not field.set_outside_value
    assert field.outside_value == -1.0
    assert isinstance(field.path, Path)
    assert field.path.exists()

    mesh = mesher.generate_geodataframe()
    assert_refinement(mesh)

    mesher.clear_fields()
    mesher.add_structured_field(
        cellsize=cellsize,
        xmin=x.min(),
        ymin=y.min(),
        dx=1.0,
        dy=1.0,
        outside_value=100.0,
    )
    field = mesher.fields[0]
    assert isinstance(field, gf.StructuredField)
    assert field.set_outside_value
    assert field.outside_value == 100.0
    assert isinstance(field.path, Path)
    assert field.path.exists()


def test_add_structured_field_from_dataarray(gdf):
    mesher = GmshMesher._force_init(gdf)

    with pytest.raises(
        TypeError, match="da must be xr.DataArray, received instead: int"
    ):
        mesher.add_structured_field_from_dataarray(1)

    x = np.arange(1.0, 10.0)
    y = np.arange(1.0, 10.0)
    da = xr.DataArray(
        np.ones((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x")
    )

    with pytest.raises(ValueError, match=re.escape('Dimensions must be ("y", "x")')):
        mesher.add_structured_field_from_dataarray(da.transpose())

    with pytest.raises(ValueError, match=r"Minimum cellsize must be > 0, received:"):
        mesher.add_structured_field_from_dataarray(da * -1)

    da_x = da.copy()
    da_x["x"] = da_x["x"] ** 2

    with pytest.raises(ValueError, match="da is not equidistant along x"):
        mesher.add_structured_field_from_dataarray(da_x)

    da_y = da.copy()
    da_y["y"] = da_y["y"] ** 2

    with pytest.raises(ValueError, match="da is not equidistant along y"):
        mesher.add_structured_field_from_dataarray(da_y)
