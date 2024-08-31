import pathlib
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Tuple, Union

import geopandas as gpd
import numpy as np

from pandamesh.common import (
    FloatArray,
    IntArray,
    central_origin,
    check_geodataframe,
    gmsh,
    move_origin,
    repr,
    separate,
)
from pandamesh.gmsh_enums import (
    FieldCombination,
    GeneralVerbosity,
    MeshAlgorithm,
    SubdivisionAlgorithm,
)
from pandamesh.gmsh_fields import (
    CombinationField,
    DistanceField,
    MathEvalField,
    StructuredField,
    ThresholdField,
)
from pandamesh.gmsh_geometry import add_distance_geometry, add_geometry
from pandamesh.mesher_base import MesherBase


@contextmanager
def gmsh_env(read_config_files: bool = True, interruptible: bool = True):
    try:
        gmsh.initialize(
            readConfigFiles=read_config_files, run=False, interruptible=interruptible
        )
        # There's gotta be a better way to actually raise proper errors...
        gmsh.option.setNumber("General.Terminal", 1)
        yield
    finally:
        gmsh.finalize()


class GmshMesher(MesherBase):
    """
    Wrapper for the python bindings to Gmsh. This class must be initialized
    with a geopandas GeoDataFrame containing at least one polygon, and a column
    named ``"cellsize"``.

    Optionally, multiple polygons with different cell sizes can be included in
    the geodataframe. These can be used to achieve local mesh refinement.

    Linestrings and points may also be included. The segments of linestrings
    will be directly forced into the triangulation. Points can also be forced
    into the triangulation. Unlike Triangle, the cell size values associated
    with these geometries **will** be used.

    Gmsh cannot automatically resolve overlapping polygons, or points
    located exactly on segments. During initialization, the geometries of
    the geodataframe are checked:

        * Polygons should not have any overlap with each other.
        * Linestrings should not intersect each other, unless the intersection
          vertex is present in both.
        * Every linestring should be fully contained by a single polygon;
          a linestring may not intersect two or more polygons.
        * Linestrings and points should not "touch" / be located on
          polygon borders.
        * Holes in polygons are fully supported, but they must not contain
          any linestrings or points.

    If such cases are detected, the initialization will error: use the
    :class:`pandamesh.Preprocessor` to clean up geometries beforehand.

    For more details on Gmsh, see:
    https://gmsh.info/doc/texinfo/gmsh.html

    A helpful index can be found near the bottom:
    https://gmsh.info/doc/texinfo/gmsh.html#Syntax-index

    .. note::

        This meshers uses the Gmsh Python API, which is global. To avoid a
        situation where multiple GmshMeshers have been iniatilized and are
        mutating each other's (global) variables, the ``.finalize()`` method
        must be called before instantiating a new mesher.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame containing the vector geometry. Must contain a "cellsize"
        column.
    shift_origin: bool, optional, default is True.
        If True, temporarily shifts the coordinate system origin to the centroid
        of the geometry's bounding box during mesh generation. This helps mitigate
        floating-point precision issues. The resulting mesh vertices are
        automatically translated back to the original coordinate system.
    read_config_files: bool
        Gmsh initialization option: Read system Gmsh configuration files
        (gmshrc and gmsh-options).
    interruptible: bool
        Gmsh initialization option.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "GmshMesher":
        # The Gmsh Python API is unfortunately global. Make sure only one
        # GmshMesher instance is active at a given time.
        if cls._instance is not None:
            raise RuntimeError(
                f"Singleton class {cls.__name__} is already instantiated. "
                "Please call .finalize() on other GmshMesher instance prior to "
                "initializing a new one."
            )
        cls._instance = super().__new__(cls)
        cls._lock.acquire()
        cls._instance._initialized = True
        return cls._instance

    @classmethod
    def finalize(cls) -> None:
        # Finalize gmsh, release locks.
        try:
            cls.finalize_gmsh()
        finally:
            if cls._lock.locked():
                cls._lock.release()
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None

    def __getattribute__(self, name: str) -> Any:
        # Make sure to error if finalized has already been called.
        if name in ("__dict__", "_initialized", "_initialize_gmsh", "finalize"):
            return super().__getattribute__(name)
        if not self._initialized:
            raise RuntimeError("GmshMesher has been finalized")
        return super().__getattribute__(name)

    @classmethod
    def get_instance(
        cls,
        gdf: gpd.GeoDataFrame,
        shift_origin: bool = True,
        read_config_files: bool = True,
        interruptible: bool = True,
    ) -> "GmshMesher":
        """
        Guarantees a new instance.

        This method ensures that only one instance of GmshMesher exists at a
        time. If an instance already exists, it is finalized before this method
        creates a new one.
        """
        cls.finalize()
        return cls(gdf, shift_origin, read_config_files, interruptible)

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        shift_origin: bool = True,
        read_config_files: bool = True,
        interruptible: bool = True,
    ) -> None:
        self._initialize_gmsh(
            read_config_files=read_config_files, interruptible=interruptible
        )
        check_geodataframe(gdf, {"geometry", "cellsize"}, check_index=True)
        gdf, self._xoff, self._yoff = central_origin(gdf, shift_origin)
        polygons, linestrings, points = separate(gdf)

        # Include geometry into gmsh
        add_geometry(polygons, linestrings, points)

        # Initialize fields parameters
        self._fields = []
        self._combination_field = None
        self._tmpdir = tempfile.TemporaryDirectory()

        # Set default values for meshing parameters
        self.mesh_algorithm = MeshAlgorithm.AUTOMATIC
        self.recombine_all = False
        self.mesh_size_extend_from_boundary = True
        self.mesh_size_from_points = True
        self.mesh_size_from_curvature = False
        self.field_combination = FieldCombination.MIN
        self.subdivision_algorithm = SubdivisionAlgorithm.NONE
        self.general_verbosity = GeneralVerbosity.SILENT

    def __repr__(self):
        return repr(self)

    @staticmethod
    def _initialize_gmsh(read_config_files: bool = True, interruptible: bool = True):
        GmshMesher.finalize_gmsh()
        gmsh.initialize(
            readConfigFiles=read_config_files, run=False, interruptible=interruptible
        )
        gmsh.option.setNumber("General.Terminal", 1)

    @staticmethod
    def finalize_gmsh():
        """Finalize Gmsh."""
        if gmsh.is_initialized() == 1:
            gmsh.finalize()

    # Properties
    # ----------
    @property
    def mesh_algorithm(self) -> MeshAlgorithm:
        """
        Can be set to one of :py:class:`pandamesh.MeshAlgorithm`:

        .. code::

            MESH_ADAPT = 1
            AUTOMATIC = 2
            INITIAL_MESH_ONLY = 3
            FRONTAL_DELAUNAY = 5
            BAMG = 7
            FRONTAL_DELAUNAY_FOR_QUADS = 8
            PACKING_OF_PARALLELLOGRAMS = 9
            QUASI_STRUCTURED_QUAD = 11

        Each algorithm has its own advantages and disadvantages.
        """
        return MeshAlgorithm(gmsh.option.getNumber("Mesh.Algorithm"))

    @mesh_algorithm.setter
    def mesh_algorithm(self, value: MeshAlgorithm):
        value = MeshAlgorithm.from_value(value)
        gmsh.option.setNumber("Mesh.Algorithm", value.value)

    @property
    def recombine_all(self) -> bool:
        """
        Apply recombination algorithm to all surfaces, ignoring per-surface
        spec.
        """
        return self._recombine_all

    @recombine_all.setter
    def recombine_all(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("recombine_all must be a bool")
        self._recombine_all = value
        gmsh.option.setNumber("Mesh.RecombineAll", value)

    @property
    def mesh_size_extend_from_boundary(self) -> bool:
        """
        Forces the mesh size to be extended from the boundary, or not, per
        surface.
        """
        return self._mesh_size_extend_from_boundary

    @mesh_size_extend_from_boundary.setter
    def mesh_size_extend_from_boundary(self, value):
        if not isinstance(value, bool):
            raise TypeError("mesh_size_extend_from_boundary must be a bool")
        self._mesh_size_extend_from_boundary = value
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", value)

    @property
    def mesh_size_from_points(self):
        """Compute mesh element sizes from values given at geometry points."""
        return self._mesh_size_from_points

    @mesh_size_from_points.setter
    def mesh_size_from_points(self, value):
        if not isinstance(value, bool):
            raise TypeError("mesh_size_from_points must be a bool")
        self._mesh_size_from_points = value
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", value)

    @property
    def mesh_size_from_curvature(self):
        """
        Automatically compute mesh element sizes from curvature, using the value as
        the target number of elements per 2 * Pi radians.
        """
        return self._mesh_size_from_curvature

    @mesh_size_from_curvature.setter
    def mesh_size_from_curvature(self, value):
        """
        Automatically compute mesh element sizes from curvature, using the
        value as the target number of elements per 2 * Pi radians.
        """
        if not isinstance(value, bool):
            raise TypeError("mesh_size_from_curvature must be a bool")
        self._mesh_size_from_curvature = value
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", value)

    @property
    def field_combination(self) -> FieldCombination:
        """
        Controls how cell size fields are combined when they are found at the
        same location. Can be set to one of
        :py:class:`pandamesh.FieldCombination`:

        .. code::

            MIN = "Min"
            MAX = "Max"
            MEAN = "Mean"

        """
        return self._field_combination

    @field_combination.setter
    def field_combination(self, value: Union[FieldCombination, str]):
        value = FieldCombination.from_value(value)
        self._field_combination = value
        # Value is propagated to gmsh in ._combine_fields()

    @property
    def subdivision_algorithm(self) -> SubdivisionAlgorithm:
        """
        All meshes can be subdivided to generate fully quadrangular cells. Can
        be set to one of :py:class:`pandamesh.SubdivisionAlgorithm`:

        .. code::

            NONE = 0
            ALL_QUADRANGLES = 1
            BARYCENTRIC = 3

        """

        return SubdivisionAlgorithm(gmsh.option.getNumber("Mesh.SubdivisionAlgorithm"))

    @subdivision_algorithm.setter
    def subdivision_algorithm(self, value: Union[SubdivisionAlgorithm, str]):
        value = SubdivisionAlgorithm.from_value(value)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", value.value)

    @property
    def general_verbosity(self) -> GeneralVerbosity:
        """
        Controls level of information printed. Can be set to one of
        :py:class:`pandamesh.GeneralVerbosity`:

        .. code::

            SILENT = 0
            ERRORS = 1
            WARNINGS = 2
            DIRECT = 3
            INFORMATION = 4
            STATUS = 5
            DEBUG = 99

        """
        return GeneralVerbosity(gmsh.option.getNumber("General.Verbosity"))

    @general_verbosity.setter
    def general_verbosity(self, value: Union[GeneralVerbosity, str]) -> None:
        value = GeneralVerbosity.from_value(value)
        gmsh.option.setNumber("General.Verbosity", value.value)

    @property
    def fields(self):
        """
        Read-only access to fields.

        Use ``.clear_fields`` to remove fields from the mesher.
        """
        return self._fields.copy()

    # Methods
    # -------

    def clear_fields(self):
        while self._fields:
            field = self._fields.pop()
            field.remove_from_gmsh()
        if self._combination_field is not None:
            self._combination_field.remove_from_gmsh()
            self._combination_field = None

    def add_matheval_distance_field(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Add a matheval distance field to the mesher.

        The of geometry of these fields are not forced into the mesh, but they
        are used to specify zones of with cell sizes.

        Uses the MathEval functionality in Gmsh, which relies on the SSCILIB
        math expression evaluator.

        https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/contrib/MathEx/mathex.cpp

        https://sscilib.sourceforge.net/

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Location of the features to measure distance to. Should contain
            ``spacing`` column to specify the spacing of interpolated vertices
            along linestrings and polygon boundaries, and a ``function`` column
            to specify the function to control cell size as a function of
            distance. See the examples.

        Examples
        --------
        Generate a number of points:

        >>> x = np.arange(0.0, 10.0)
        >>> y = np.arange(0.0, 10.0)
        >>> points = gpd.points_from_xy(x, y)
        >>> field = gpd.GeoDataFrame(geometry=points)

        Add spacing (dummy value for points) and a function:

        >>> field["spacing"] = np.nan
        >>> field["function"] = "max(distance^2, 1.0)"

        Apply it:

        >>> mesher.add_matheval_distance_field(field)

        Operators
        =========

        The following mathematical operators are supported:

        Basic Operators
        ---------------

        - Arithmetic: ``+``, ``-``, ``*``, ``/``, ``%`` (modulo), ``^`` (power)
        - Comparison: ````<``, ``>``

        Mathematical Functions
        ----------------------

        - Absolute value: ``abs(x)``
        - Square root: ``sqrt(x)``
        - Exponential: ``exp(x)``
        - Natural logarithm: ``log(x)``
        - Base-10 logarithm: ``log10(x)``
        - Power: ``pow(x,y)``

        Statistical Functions
        ---------------------

        - Minimum: ``min(x, y, ...)``
        - Maximum: ``max(x, y, ...)``
        - Sum: ``sum(x, y, ...)``
        - Average: ``med(x, y, ...)```

        Trigonometric Functions
        -----------------------

        - Standard: ``sin(x)``, ``cos(x)``, ``tan(x)``
        - Inverse: ``asin(x)``, ``acos(x)``, ``atan(x)``
        - Hyperbolic: ``sinh(x)``, ``cosh(x)``, ``tanh(x)``

        Rounding Functions
        ------------------

        - ``floor(x)``, ``ceil(x)``, ``round(x)``, ``trunc(x)``

        Constants
        ---------

        - Pi: ``pi``
        - Euler's number: ``e``
        """
        check_geodataframe(gdf, {"geometry", "function", "spacing"})
        gdf = move_origin(gdf, xoff=self._xoff, yoff=self._yoff)

        for function, group in gdf.groupby("function"):
            point_tags = add_distance_geometry(group["geometry"], group["spacing"])
            distance_field = DistanceField(point_tags)
            math_eval_field = MathEvalField(distance_field, function)
            self._fields.extend((distance_field, math_eval_field))
        return

    def add_threshold_distance_field(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Add a distance field to the mesher.

        The of geometry of these fields are not forced into the mesh, but they
        can be used to specify zones of with cell sizes.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Location of the features to measure distance to. Should contain
            ``spacing`` column to specify the spacing of interpolated vertices
            along linestrings and polygon boundaries.
        """
        columns = ["size_min", "size_max", "dist_min", "dist_max"]
        check_geodataframe(gdf, set(columns + ["geometry", "spacing"]))
        gdf = move_origin(gdf, xoff=self._xoff, yoff=self._yoff)

        if "sigmoid" not in gdf.columns:
            gdf["sigmoid"] = False
        if "stop_at_dist_max" not in gdf.columns:
            gdf["stop_at_dist_max"] = False
        columns += ["sigmoid", "stop_at_dist_max"]

        grouped = gdf.groupby(columns)
        for (
            size_min,
            size_max,
            dist_min,
            dist_max,
            sigmoid,
            stop_at_dist_max,
        ), group in grouped:
            point_tags = add_distance_geometry(group["geometry"], group["spacing"])
            distance_field = DistanceField(point_tags)
            threshold_field = ThresholdField(
                distance_field=distance_field,
                size_min=size_min,
                size_max=size_max,
                dist_min=dist_min,
                dist_max=dist_max,
                sigmoid=sigmoid,
                stop_at_dist_max=stop_at_dist_max,
            )
            self._fields.extend((distance_field, threshold_field))

        return

    def add_structured_field(
        self,
        cellsize: FloatArray,
        xmin: float,
        ymin: float,
        dx: float,
        dy: float,
        outside_value: Union[float, None] = None,
    ) -> None:
        """
        Add an equidistant structured field specifying cell sizes. Gmsh will
        interpolate between the points to determine the desired cell size.

        Parameters
        ----------
        cellsize: FloatArray with shape ``(n_y, n_x``)
            Specifies the cell size on a structured grid. The location of this grid
            is determined by ``xmin, ymin, dx, dy``.
        xmin: float
            x-origin.
        ymin: float
            y-origin.
        dx: float
            Spacing along the x-axis.
        dy: float
            Spacing along the y-axis.
        outside_value: Union[float, None]
            Value outside of the window ``(xmin, xmax)`` and ``(ymin, ymax)``.
            Default value is None.
        """
        self._fields.append(
            StructuredField(
                tmpdir=self._tmpdir,
                cellsize=cellsize,
                xmin=xmin - self._xoff,
                ymin=ymin - self._yoff,
                dx=dx,
                dy=dy,
                outside_value=outside_value,
            )
        )

    def add_structured_field_from_dataarray(
        self,
        da: "xarray.DataArray",  # noqa: F821
        outside_value: Union[float, None] = None,
    ):
        """
        Add an equidistant structured field specifying cell sizes from an
        xarray DataArray object. Gmsh will interpolate between the points to
        determine the desired cell size.

        Parameters
        ----------
        da: xarray.DataArray
            Values are used as cell sizes. Must have dimensions `("y", "x")`.
        outside_value: Union[float, None]
            Value outside of the window ``(xmin, xmax)`` and ``(ymin, ymax)``.
            Default value is None.
        """
        import xarray as xr

        if not isinstance(da, xr.DataArray):
            raise TypeError(
                f"da must be xr.DataArray, received instead: {type(da).__name__}"
            )
        if not da.dims == ("y", "x"):
            raise ValueError(f'Dimensions must be ("y", "x"), received: {da.dims}')

        dxs = np.diff(da["x"]) if len(da["x"]) > 0 else np.array([0.0])
        dys = np.diff(da["y"]) if len(da["y"]) > 0 else np.array([0.0])
        dx = dxs[0]
        dy = dys[0]

        if not np.allclose(dxs, dx, atol=1e-4 * dx):
            raise ValueError("da is not equidistant along x")
        if not np.allclose(dys, dx, atol=1e-4 * dy):
            raise ValueError("da is not equidistant along y")

        self.add_structured_field(
            cellsize=da.to_numpy(),
            xmin=float(da["x"].min()),
            ymin=float(da["y"].min()),
            dx=dx,
            dy=dy,
            outside_value=outside_value,
        )

    def _combine_fields(self) -> None:
        if self._combination_field is not None:
            self._combination_field.remove_from_gmsh()
        self._combination_field = CombinationField(self._fields, self.field_combination)

    def _vertices(self):
        # getNodes returns: node_tags, coord, parametric_coord
        _, vertices, _ = gmsh.model.mesh.getNodes()
        # Return x and y
        vertices = vertices.reshape((-1, 3))[:, :2]
        vertices[:, 0] += self._xoff
        vertices[:, 1] += self._yoff
        return np.ascontiguousarray(vertices)

    def _faces(self):
        element_types, _, node_tags = gmsh.model.mesh.getElements()
        tags = dict(zip(element_types, node_tags))
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
        """
        Generate a mesh of triangles or quadrangles.

        Returns
        -------
        vertices: np.ndarray of floats with shape ``(n_vertex, 2)``
        faces: np.ndarray of integers with shape ``(n_face, nmax_per_face)``
            ``nmax_per_face`` is 3 for exclusively triangles and 4 if
            quadrangles are included. A fill value of -1 is used as a last
            entry for triangles in that case.
        """
        # Remove any previously generated results from gmsh.
        gmsh.model.mesh.clear()
        self._combine_fields()
        gmsh.model.mesh.generate(dim=2)

        # cleaning up of mesh in order to obtain unique elements and nodes
        gmsh.model.mesh.removeDuplicateElements()
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.renumberElements()
        gmsh.model.mesh.renumberNodes()

        return self._vertices(), self._faces()

    def write(self, path: Union[str, pathlib.Path]):
        """
        Write a gmsh .msh file

        Parameters
        ----------
        path: Union[str, pathlib.Path
        """
        gmsh.write(str(path))
