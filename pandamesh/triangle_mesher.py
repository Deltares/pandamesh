from enum import Enum
from typing import Tuple, Union

import geopandas as gpd
import triangle

from pandamesh.common import (
    FloatArray,
    IntArray,
    check_geodataframe,
    invalid_option,
    repr,
    separate,
    to_ugrid,
)
from pandamesh.triangle_geometry import collect_geometry, polygon_holes


class DelaunayAlgorithm(Enum):
    DIVIDE_AND_CONQUER = ""
    INCREMENTAL = "i"
    SWEEPLINE = "F"


class TriangleMesher:
    """
    Wrapper for the python bindings to Triangle. This class must be initialized
    with a geopandas GeoDataFrame containing at least one polygon, and a column
    named ``"cellsize"``.

    Optionally, multiple polygons with different cell sizes can be included in
    the geodataframe. These can be used to achieve local mesh remfinement.

    Linestrings and points may also be included. The segments of linestrings
    will be directly forced into the triangulation. Points can also be forced
    into the triangulation. The cell size values associated with these
    geometries willl not be used.

    Triangle cannot automatically resolve overlapping polygons, or points
    located exactly on segments. During initialization, the geometries of
    the geodataframe are checked:

        * Polygons should not have any overlap with each other.
        * Linestrings should not intersect each other.
        * Every linestring should be fully contained by a single polygon;
          a linestring may not intersect two or more polygons.
        * Linestrings and points should not "touch" / be located on
          polygon borders.
        * Holes in polygons are fully supported, but they must not contain
          any linestrings or points.

    If such cases are detected, the initialization will error.

    For more details on Triangle, see:
    https://www.cs.cmu.edu/~quake/triangle.defs.html
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        check_geodataframe(gdf)
        polygons, linestrings, points = separate(gdf)
        self.vertices, self.segments, self.regions = collect_geometry(
            polygons, linestrings, points
        )
        self.holes = polygon_holes(polygons)

        # Set default values for meshing parameters
        self.minimum_angle = 20.0
        self.conforming_delaunay = True
        self.suppress_exact_arithmetic = False
        self.maximum_steiner_points = None
        self.delaunay_algorithm = DelaunayAlgorithm.DIVIDE_AND_CONQUER
        self.consistency_check = False

    def __repr__(self):
        return repr(self)

    @property
    def minimum_angle(self) -> float:
        """
        Minimum allowed angle for any triangle in the mesh.

        See:
        https://www.cs.cmu.edu/~quake/triangle.q.html
        """
        return self._minimum_angle

    @minimum_angle.setter
    def minimum_angle(self, value: float):
        if not isinstance(value, float):
            raise TypeError("minimum angle must be a float")
        if value >= 34.0 or value <= 0.0:
            raise ValueError("minimum_angle should fall in the interval: (0.0, 34.0)")
        self._minimum_angle = value

    @property
    def conforming_delaunay(self) -> bool:
        """
        Conforming Delaunay: use this switch if you want all triangles in the
        mesh to be Delaunay, and not just constrained Delaunay; or if you want
        to ensure that all Voronoi vertices lie within the triangulation.
        """
        return self._conforming_delaunay

    @conforming_delaunay.setter
    def conforming_delaunay(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("conforming_delaunay must be a bool")
        self._conforming_delaunay = value

    @property
    def suppress_exact_arithmetic(self) -> bool:
        """
        Suppresses exact arithmetic.

        See:
        https://www.cs.cmu.edu/~quake/triangle.exact.html
        """
        return self._suppress_exact_arithmetic

    @suppress_exact_arithmetic.setter
    def suppress_exact_arithmetic(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("suppress_exact_arithmetic must be a bool")
        self._suppress_exact_arithmetic = value

    @property
    def maximum_steiner_points(self) -> int:
        """
        Specifies the maximum number of added Steiner points

        See:
        https://www.cs.cmu.edu/~quake/triangle.S.html
        """
        return self._maximum_steiner_points

    @maximum_steiner_points.setter
    def maximum_steiner_points(self, value: Union[int, None]):
        if not isinstance(value, (int, type(None))):
            raise TypeError("maximum_steiner_points must be an int or None")
        self._maximum_steiner_points = value

    @property
    def delaunay_algorithm(self) -> DelaunayAlgorithm:
        """
        ``DelaunayAlgoritm.DIVIDE_AND_CONQUER``: Default algorithm.

        ``DelaunayAlgoritm.INCREMENTAL``: Uses the incremental algorithm for
        Delaunay triangulation, rather than the divide-and-conquer algorithm.

        ``DelaunayAlgoritm.SWEEPLINE``: Uses Steven Fortune’s sweepline
        algorithm for Delaunay triangulation, rather than the
        divide-and-conquer algorithm.
        """
        return self._delaunay_algorithm

    @delaunay_algorithm.setter
    def delaunay_algorithm(self, value: DelaunayAlgorithm):
        if value not in DelaunayAlgorithm:
            raise ValueError(invalid_option(value, DelaunayAlgorithm))
        self._delaunay_algorithm = value

    @property
    def consistency_check(self) -> bool:
        """
        Check the consistency of the final mesh. Uses exact arithmetic for
        checking, even if ``suppress_exact_arithmetic`` is set to ``False``.
        Useful if you suspect Triangle is buggy.
        """
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self, value: bool):
        if not isinstance(value, int):
            raise TypeError("consistency_check must be a bool")
        self._consistency_check = value

    def generate(self) -> Tuple[FloatArray, IntArray]:
        """
        Generate a mesh of triangles.

        Returns
        -------
        vertices: np.ndarray of floats with shape ``(n_vertex, 2)``
        triangles: np.ndarray of integers with shape ``(n_triangle, 3)``
        """
        options = (
            "p"
            f"q{self._minimum_angle}"
            "a"
            f"{'D' if self._conforming_delaunay else ''}"
            f"{'X' if self._suppress_exact_arithmetic else ''}"
            f"{'S' + str(self._maximum_steiner_points) if self._maximum_steiner_points is not None else ''}"
            f"{self._delaunay_algorithm.value}"
            f"{'C' if self.consistency_check else ''}"
        )

        tri = {"vertices": self.vertices, "segments": self.segments}
        if self.holes is not None:
            tri["holes"] = self.holes
        if len(self.regions) > 0:
            tri["regions"] = self.regions

        result = triangle.triangulate(tri=tri, opts=options)
        return result["vertices"], result["triangles"]

    def generate_ugrid(self) -> "xugrid.Ugrid2d":  # type: ignore # noqa
        return to_ugrid(*self.generate())
