import pathlib
import struct
from typing import Union

import numpy as np

from pandamesh.common import FloatArray, gmsh, repr
from pandamesh.gmsh_enums import FieldCombination


def write_structured_field_file(
    path: Union[pathlib.Path, str],
    cellsize: FloatArray,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
) -> None:
    """
    Write a binary structured 2D gmsh field file.

    Note: make sure the signs of ``dx`` and ``dy`` match the orientation of the
    data in ``cellsize``. Geospatial rasters typically have a positive value for
    dx and negative for dy (x coordinate is ascending; y coordinate is
    descending). Data will be flipped around the respective axis for a negative
    dx or dy.

    Parameters
    ----------
    path: str or pathlib.Path
    cellsize: 2D np.ndarray of floats
        Dimension order is (y, x), i.e. y differs along the rows and x differs along
        the columns.
    xmin: float
    ymin: float
    dx: float
    dy: float

    Returns
    -------
    None
        Writes a structured gmsh field file.
    """
    shape = cellsize.shape
    if cellsize.ndim != 2:
        raise ValueError(f"`cellsize` must be 2D. Received an array of shape: {shape}")
    nrow, ncol = shape
    # Flip values around if dx or dy is negative.
    if dy < 0.0:
        cellsize = np.flipud(cellsize)
        dy = abs(dy)
    if dx < 0.0:
        cellsize = np.fliplr(cellsize)
        dx = abs(dx)

    with open(path, "wb") as f:
        f.write(struct.pack("3d", xmin, ymin, 0.0))
        f.write(struct.pack("3d", dx, dy, 1.0))
        f.write(struct.pack("3i", nrow, ncol, 1))
        cellsize.tofile(f)
    return


def add_math_eval_field(field: dict, distance_id: int, field_id: int) -> None:
    function = field["function"]
    if "{distance}" not in function:
        raise ValueError("{distance} not in MathEval field function")
    gmsh.model.mesh.field.add("MathEval", field_id)
    distance = f"F{distance_id}"
    gmsh.model.mesh.field.setString(field_id, "F", function.format(distance=distance))


def add_threshold_field(
    field: dict,
    field_id: int,
    distance_id: int,
) -> None:
    gmsh.model.mesh.field.add("Threshold", field_id)
    gmsh.model.mesh.field.setNumber(field_id, "IField", distance_id)
    gmsh.model.mesh.field.setNumber(field_id, "LcMin", field["lc_min"])
    gmsh.model.mesh.field.setNumber(field_id, "LcMax", field["lc_max"])
    gmsh.model.mesh.field.setNumber(field_id, "DistMin", field["dist_min"])
    gmsh.model.mesh.field.setNumber(field_id, "DistMax", field["dist_max"])
    gmsh.model.mesh.field.setNumber(
        field_id, "StopAtDistMax", field["stop_at_dist_max"]
    )
    gmsh.model.mesh.field.setNumber(field_id, "Sigmoid", field["sigmoid"])
    return


class GmshField:
    def remove_from_gmsh(self):
        gmsh.model.mesh.field.remove(self.id)

    def __repr__(self) -> str:
        return repr(self)


class DistanceField(GmshField):
    def __init__(self, point_tags):
        self.id = gmsh.model.mesh.field.add("Distance")
        self.point_list = point_tags
        gmsh.model.mesh.field.setNumbers(self.id, "PointsList", self.point_list)


class MathEvalField(GmshField):
    def __init__(self, distance_field: DistanceField, function: str):
        if "distance" not in function:
            raise ValueError(f"distance not in MathEval field function: {function}")
        self.id = gmsh.model.mesh.field.add("MathEval")
        self.distance_field_id = distance_field.id
        self.function = function
        distance_function = function.replace("distance", f"F{self.distance_field_id}")
        gmsh.model.mesh.field.setString(self.id, "F", distance_function)


class ThresholdField(GmshField):
    def __init__(
        self,
        distance_field: DistanceField,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
        sigmoid: bool = False,
        stop_at_dist_max: bool = False,
    ):
        self.id = gmsh.model.mesh.field.add("Threshold")
        self.distance_field_id = distance_field.id
        gmsh.model.mesh.field.setNumber(self.id, "InField", self.distance_field_id)
        gmsh.model.mesh.field.setNumber(self.id, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(self.id, "SizeMax", size_max)
        gmsh.model.mesh.field.setNumber(self.id, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(self.id, "DistMax", dist_max)
        gmsh.model.mesh.field.setNumber(self.id, "Sigmoid", sigmoid)
        gmsh.model.mesh.field.setNumber(self.id, "StopAtDistMax", stop_at_dist_max)


class StructuredField(GmshField):
    def __init__(
        self,
        tmpdir,
        cellsize: FloatArray,
        xmin: float,
        ymin: float,
        dx: float,
        dy: float,
        outside_value: Union[float, None] = None,
    ):
        if outside_value is not None:
            set_outside_value = True
        else:
            set_outside_value = False
            outside_value = -1.0

        self.id = gmsh.model.mesh.field.add("Structured")
        self.path = f"{tmpdir.name}/structured_field_{self.id}.dat"
        write_structured_field_file(self.path, cellsize, xmin, ymin, dx, dy)
        gmsh.model.mesh.field.setNumber(self.id, "TextFormat", 0)  # binary
        gmsh.model.mesh.field.setString(self.id, "FileName", self.path)
        gmsh.model.mesh.field.setNumber(self.id, "SetOutsideValue", set_outside_value)
        gmsh.model.mesh.field.setNumber(self.id, "OutsideValue", outside_value)


class CombinationField(GmshField):
    def __init__(self, fields, combination: FieldCombination):
        self.id = gmsh.model.mesh.field.add(combination.value)
        self.field_list = [field.id for field in fields]
        gmsh.model.mesh.field.setNumbers(self.id, "FieldsList", self.field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(self.id)
