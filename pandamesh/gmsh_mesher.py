import json
import pathlib
import tempfile
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import List, Tuple, Union

import geopandas as gpd
import gmsh
import numpy as np
import pandas as pd

from .common import FloatArray, IntArray, check_geodataframe, invalid_option, separate
from .gmsh_fields import FIELDS, add_distance_field, add_structured_field, validate
from .gmsh_geometry import add_field_geometry, add_geometry


@contextmanager
def gmsh_env():
    try:
        gmsh.initialize()
        # There's gotta be a better way to actually raise proper errors...
        gmsh.option.setNumber("General.Terminal", 1)
        yield
    finally:
        gmsh.finalize()


class MeshAlgorithm(IntEnum):
    MESH_ADAPT = 1
    AUTOMATIC = 2
    INITIAL_MESH_ONLY = 3
    FRONTAL_DELAUNAY = 5
    BAMG = 7
    FRONTAL_DELAUNAY_FOR_QUADS = 8
    PACKING_OF_PARALLELLOGRAMS = 9


class SubdivisionAlgorithm(IntEnum):
    NONE = 0
    ALL_QUADRANGLES = 1
    BARYCENTRIC = 3


class FieldCombination(Enum):
    MIN = "Min"
    MAX = "Max"
    MEAN = "Mean"


def coerce_field(field: Union[dict, str]) -> dict:
    if not isinstance(field, (dict, str)):
        raise TypeError("field must be a dictionary or a valid JSON dictionary string")
    if isinstance(str):
        field = json.loads(field)[0]
    return field


class GmshMesher:
    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        gmsh.clear()
        check_geodataframe(gdf)
        polygons, linestrings, points = separate(gdf)

        # Include geometry into gmsh
        add_geometry(polygons, linestrings, points)

        # Initialize fields parameters
        self._current_field_id = 0
        self._fields_list: List[int] = []
        self._distance_fields_list: List[int] = []
        self.fields = gpd.GeoDataFrame()
        self._tmpdir = tempfile.TemporaryDirectory()

        # Set default values for meshing parameters
        self.mesh_algorithm = MeshAlgorithm.AUTOMATIC
        self.recombine_all = False
        # self.force_geometry = True  # not implemented yet, see below
        self.characteristic_length_from_boundary = True
        self.characteristic_length_from_points = True
        self.characteristic_length_from_curvature = False
        self.field_combination = FieldCombination.MIN
        self.subdivision_algorithm = SubdivisionAlgorithm.NONE

    # Properties
    # ----------

    @property
    def mesh_algorithm(self):
        return gmsh.option.getNumber("Mesh.Algorithm")

    @mesh_algorithm.setter
    def mesh_algorithm(self, value: MeshAlgorithm):
        if value not in MeshAlgorithm:
            raise ValueError(invalid_option(value, MeshAlgorithm))
        gmsh.option.setNumber("Mesh.Algorithm", value.value)

    @property
    def recombine_all(self):
        return self._recombine_all

    @recombine_all.setter
    def recombine_all(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("recombine_all must be a bool")
        self._recombine_all = value
        gmsh.option.setNumber("Mesh.RecombineAll", value)

    @property
    def force_geometry(self) -> None:
        raise NotImplementedError
        # return self._force_geometry

    @force_geometry.setter
    def force_geometry(self, value: bool) -> None:
        # Wait for the next release incorporating this change:
        # https://gitlab.onelab.info/gmsh/gmsh/-/merge_requests/358
        raise NotImplementedError
        # if not isinstance(value, bool):
        #     raise TypeError("force_geometry must be a bool")
        # self._force_geometry = value

    @property
    def characteristic_length_from_boundary(self):
        return self._characteristic_length_from_boundary

    @characteristic_length_from_boundary.setter
    def characteristic_length_from_boundary(self, value):
        if not isinstance(value, bool):
            raise TypeError("characteristic_length_from_boundary must be a bool")
        self._characteristic_length_from_boundary = value
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", value)

    @property
    def characteristic_length_from_points(self):
        return self._characteristic_length_from_points

    @characteristic_length_from_points.setter
    def characteristic_length_from_points(self, value):
        if not isinstance(value, bool):
            raise TypeError("characteristic_length_from_boundary must be a bool")
        self._characteristic_length_from_points = value
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", value)

    @property
    def characteristic_length_from_curvature(self):
        return self._characteristic_length_from_curvature

    @characteristic_length_from_curvature.setter
    def characteristic_length_from_curvature(self, value):
        if not isinstance(value, bool):
            raise TypeError("characteristic_length_from_curvature must be a bool")
        self._characteristic_length_from_curvature = value
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", value)

    @property
    def field_combination(self):
        return self._field_combination

    @field_combination.setter
    def field_combination(self, value):
        if value not in FieldCombination:
            raise ValueError(invalid_option(value, FieldCombination))
        self._field_combination = value

    @property
    def subdivision_algorithm(self):
        self._subdivision_algorithm

    @subdivision_algorithm.setter
    def subdivision_algorithm(self, value):
        if value not in SubdivisionAlgorithm:
            raise ValueError(invalid_option(value, SubdivisionAlgorithm))
        self._subdivision_algorithm = value
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", value)

    # Methods
    # -------

    def _new_field_id(self) -> int:
        self._current_field_id += 1
        return self._current_field_id

    def _combine_fields(self) -> None:
        # Create a combination field
        if self.fields is None:
            return
        field_id = self._new_field_id()
        gmsh.model.mesh.field.add(self.field_combination.value, field_id)
        gmsh.model.mesh.field.setNumbers(field_id, "FieldsList", self._fields_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

    def clear_fields(self) -> None:
        self.fields = None
        for field_id in self._fields_list + self._distance_fields_list:
            gmsh.model.mesh.field.remove(field_id)
        self._fields_list = []
        self._distance_fields_list = []
        self._current_field_id = 0

    def add_distance_field(
        self, gdf: gpd.GeoDataFrame, minimum_cellsize: float
    ) -> None:
        if "field" not in gdf.columns:
            raise ValueError("field column is missing from geodataframe")

        # Explode just in case to get rid of multi-elements
        gdf = gdf.explode()

        for field, field_gdf in gdf.groupby("field"):
            distance_id = self._new_field_id()
            field_id = self._new_field_id()
            field_dict = coerce_field(field)
            fieldtype = field_dict["type"].lower()

            spec, add_field = FIELDS[fieldtype.lower()]
            try:
                validate(field_dict, spec)
            except KeyError:
                raise ValueError(
                    f'invalid field type {fieldtype}. Allowed are: "MathEval", "Threshold".'
                )

            # Insert geometry, and create distance field
            nodes_list = add_field_geometry(field_gdf, minimum_cellsize)
            add_distance_field(nodes_list, [], 0, distance_id)

            # Add field based on distance field
            add_field(field_dict, distance_id=distance_id, field_id=field_id)
            self._fields_list.append(field_id)
            self._distance_fields_list.append(distance_id)

        self.fields = pd.concat(self.fields, gdf)

    def add_structured_field(
        self,
        cellsize: FloatArray,
        xmin: float,
        ymin: float,
        dx: float,
        dy: float,
        outside_value: Union[float, None] = None,
    ) -> None:
        if outside_value is not None:
            set_outside_value = True
        else:
            set_outside_value = False
            outside_value = -1.0

        field_id = self._new_field_id()
        path = f"{self._tmpdir.name}/structured_field_{field_id}.dat"
        add_structured_field(
            cellsize,
            xmin,
            ymin,
            dx,
            dy,
            outside_value,
            set_outside_value,
            field_id,
            path,
        )
        self._fields_list.append(field_id)

    def vertices(self):
        # getNodes returns: node_tags, coord, parametric_coord
        _, vertices, _ = gmsh.model.mesh.getNodes()
        # Return x and y
        return vertices.reshape((-1, 3))[:, :2]

    def faces(self):
        element_types, _, node_tags = gmsh.model.mesh.getElements()
        tags = {etype: tags for etype, tags in zip(element_types, node_tags)}
        _TRIANGLE = 2
        _QUAD = 3
        _FILL_VALUE = 0
        # Combine triangle and quad faces if the mesh is heterogenous
        if _TRIANGLE in tags and _QUAD in tags:
            triangle_faces = tags[_TRIANGLE].reshape((-1, 3))
            quad_faces = tags[_QUAD].reshape((-1, 4))
            n_triangle = triangle_faces.shape[0]
            n_quad = quad_faces.shape[0]
            faces = np.full((n_triangle + n_quad, 4), _FILL_VALUE)
            faces[:n_triangle, :3] = triangle_faces
            faces[n_triangle:, :] = quad_faces
        elif _QUAD in tags:
            faces = tags[_QUAD].reshape((-1, 4))
        elif _TRIANGLE in tags:
            faces = tags[_TRIANGLE].reshape((-1, 3))
        else:
            raise ValueError("No triangles or quads in mesh")
        # convert to 0-based index
        return faces - 1

    def generate(self) -> Tuple[FloatArray, IntArray]:
        self._combine_fields()
        gmsh.model.mesh.generate(dim=2)
        return self.vertices(), self.faces()

    def write(self, path: Union[str, pathlib.Path]):
        """
        Writes a gmsh .msh file
        """
        gmsh.write(path)
