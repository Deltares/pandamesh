import pathlib
import struct
from typing import List, Tuple, Union

import numpy as np

from pandamesh.common import FloatArray, gmsh


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
    if cellsize.ndim() != 2:
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


def add_distance_field(
    nodes_list: List[int],
    edges_list: List[int],
    n_nodes_by_edge: int,
    field_id: int,
) -> None:
    gmsh.model.mesh.field.add("Distance", field_id)
    gmsh.model.mesh.field.setNumbers(field_id, "NodesList", nodes_list)
    gmsh.model.mesh.field.setNumbers(field_id, "NNodesByEdge", n_nodes_by_edge)
    gmsh.model.mesh.field.setNumbers(field_id, "EdgesList", edges_list)
    return None


def validate_field(field: dict, spec: List[Tuple[str, type]]) -> None:
    for key, dtype in spec:
        fieldtype = field["type"]
        if key not in field:
            raise KeyError(f'Key "{key}" is missing for field {fieldtype}')
        if not isinstance(field[key], dtype):
            raise TypeError(
                f"Entry {key} must be of type {dtype} for field {fieldtype}, "
                f"received instead: {type(field[key])}"
            )


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


def add_structured_field(
    cellsize: FloatArray,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    outside_value: float,
    set_outside_value: bool,
    field_id: int,
    path: str,
) -> None:
    write_structured_field_file(path, cellsize, xmin, ymin, dx, dy)
    gmsh.model.mesh.field.add("Structured", field_id)
    gmsh.model.mesh.field.setNumber(field_id, "TextFormat", 0)
    gmsh.model.mesh.field.setString(field_id, "FileName", path)


MATHEVAL_SPEC = [("function", str)]

THRESHOLD_SPEC = [
    ("dist_max", float),
    ("dist_min", float),
    ("lc_max", float),
    ("lc_min", float),
    ("sigmoid", bool),
]

FIELDS = {
    "matheval": (MATHEVAL_SPEC, add_math_eval_field),
    "threshold": (THRESHOLD_SPEC, add_threshold_field),
}
