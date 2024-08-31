import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from pandamesh.common import FloatArray, IntArray, gmsh
from pandamesh.gmsh_enums import FieldCombination


def write_structured_field_file(
    path: Union[Path, str],
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
        cellsize.T.tofile(f)
    return


class GmshField:
    def remove_from_gmsh(self):
        gmsh.model.mesh.field.remove(self.id)


class DistanceFunctionField(GmshField):
    def remove_from_gmsh(self):
        self.distance_field.remove_from_gmsh()
        super().remove_from_gmsh()


@dataclass
class DistanceField(GmshField):
    point_list: IntArray
    id: int = field(init=False)

    def __post_init__(self):
        self.id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(self.id, "PointsList", self.point_list)


@dataclass
class MathEvalField(DistanceFunctionField):
    distance_field: DistanceField
    function: str
    id: int = field(init=False)

    def __post_init__(self):
        if "distance" not in self.function:
            raise ValueError(
                f"distance not in MathEval field function: {self.function}"
            )
        self.id = gmsh.model.mesh.field.add("MathEval")
        distance_function = self.function.replace(
            "distance", f"F{self.distance_field.id}"
        )
        gmsh.model.mesh.field.setString(self.id, "F", distance_function)


@dataclass
class ThresholdField(DistanceFunctionField):
    distance_field: DistanceField
    size_min: float
    size_max: float
    dist_min: float
    dist_max: float
    sigmoid: bool = False
    stop_at_dist_max: bool = False
    id: int = field(init=False)

    def __post_init__(self):
        self.id = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(self.id, "InField", self.distance_field.id)
        gmsh.model.mesh.field.setNumber(self.id, "SizeMin", self.size_min)
        gmsh.model.mesh.field.setNumber(self.id, "SizeMax", self.size_max)
        gmsh.model.mesh.field.setNumber(self.id, "DistMin", self.dist_min)
        gmsh.model.mesh.field.setNumber(self.id, "DistMax", self.dist_max)
        gmsh.model.mesh.field.setNumber(self.id, "Sigmoid", self.sigmoid)
        gmsh.model.mesh.field.setNumber(self.id, "StopAtDistMax", self.stop_at_dist_max)


@dataclass
class StructuredField(GmshField):
    tmpdir: tempfile.TemporaryDirectory
    cellsize: FloatArray
    xmin: float
    ymin: float
    dx: float
    dy: float
    outside_value: Optional[float] = None
    id: int = field(init=False)
    path: str = field(init=False)
    set_outside_value: bool = field(init=False)

    def __post_init__(self):
        min_value = self.cellsize.min()
        if not (min_value > 0):  # will also catch NaN
            raise ValueError(f"Minimum cellsize must be > 0, received: {min_value}")

        if self.outside_value is not None:
            self.set_outside_value = True
            self.outside_value = self.outside_value
        else:
            self.set_outside_value = False
            self.outside_value = -1.0

        self.id = gmsh.model.mesh.field.add("Structured")
        self.path = Path(self.tmpdir.name) / f"structured_field_{self.id}.dat"
        write_structured_field_file(
            self.path, self.cellsize, self.xmin, self.ymin, self.dx, self.dy
        )
        gmsh.model.mesh.field.setNumber(self.id, "TextFormat", 0)  # binary
        gmsh.model.mesh.field.setString(self.id, "FileName", str(self.path))
        gmsh.model.mesh.field.setNumber(
            self.id, "SetOutsideValue", self.set_outside_value
        )
        gmsh.model.mesh.field.setNumber(self.id, "OutsideValue", self.outside_value)


@dataclass
class CombinationField(GmshField):
    fields: List[GmshField]
    combination: FieldCombination
    id: int = field(init=False)
    fields_list: List[int] = field(init=False)

    def __post_init__(self):
        self.id = gmsh.model.mesh.field.add(self.combination.value)
        self.fields_list = [field.id for field in self.fields]
        gmsh.model.mesh.field.setNumbers(self.id, "FieldsList", self.fields_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(self.id)
