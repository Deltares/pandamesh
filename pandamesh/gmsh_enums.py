from enum import Enum, IntEnum


class MeshAlgorithm(IntEnum):
    """
    Each algorithm has its own advantages and disadvantages.

    For all 2D unstructured algorithms a Delaunay mesh that contains all
    the points of the 1D mesh is initially constructed using a
    divide-and-conquer algorithm. Missing edges are recovered using edge
    swaps. After this initial step several algorithms can be applied to
    generate the final mesh:

    * The MeshAdapt algorithm is based on local mesh modifications. This
      technique makes use of edge swaps, splits, and collapses: long edges
      are split, short edges are collapsed, and edges are swapped if a
      better geometrical configuration is obtained.
    * The Delaunay algorithm is inspired by the work of the GAMMA team at
      INRIA. New points are inserted sequentially at the circumcenter of
      the element that has the largest adimensional circumradius. The mesh
      is then reconnected using an anisotropic Delaunay criterion.
    * The Frontal-Delaunay algorithm is inspired by the work of S. Rebay.
    * Other experimental algorithms with specific features are also
      available. In particular, Frontal-Delaunay for Quads is a variant of
      the Frontal-Delaunay algorithm aiming at generating right-angle
      triangles suitable for recombination; and BAMG allows to generate
      anisotropic triangulations.

    For very complex curved surfaces the MeshAdapt algorithm is the most robust.
    When high element quality is important, the Frontal-Delaunay algorithm should
    be tried. For very large meshes of plane surfaces the Delaunay algorithm is
    the fastest; it usually also handles complex mesh size fields better than the
    Frontal-Delaunay. When the Delaunay or Frontal-Delaunay algorithms fail,
    MeshAdapt is automatically triggered. The Automatic algorithm uses
    Delaunay for plane surfaces and MeshAdapt for all other surfaces.
    """

    MESH_ADAPT = 1
    """
    Local mesh modifications using edge swaps, splits, and collapses. Robust
    for complex curved surfaces.
    """

    AUTOMATIC = 2
    """
    Uses Delaunay for plane surfaces and MeshAdapt for all other surfaces.
    """

    INITIAL_MESH_ONLY = 3
    """Generates only the initial Delaunay triangulation."""

    FRONTAL_DELAUNAY = 5
    """Good for high element quality."""

    BAMG = 7
    """Experimental algorithm for generating anisotropic triangulations."""

    FRONTAL_DELAUNAY_FOR_QUADS = 8
    """
    Variant of Frontal-Delaunay aiming to generate right-angle triangles
    suitable for recombination.
    """

    PACKING_OF_PARALLELLOGRAMS = 9
    """Experimental algorithm for parallelogram-based mesh generation."""

    QUASI_STRUCTURED_QUAD = 11
    """
    Combines an initial unstructured quad mesh with topological improvements
    guided by cross fields to produce quasi-structured meshes with few
    irregular vertices.
    """


class SubdivisionAlgorithm(IntEnum):
    """
    The default recombination algorithm might leave some triangles in the mesh,
    if recombining all the triangles leads to badly shaped quads. In such
    cases, to generate full-quad meshes, you can either subdivide the resulting
    hybrid mesh (ALL_QUADRANGLES), or use the full-quad recombination
    algorithm, which will automatically perform a coarser mesh followed by
    recombination, smoothing and subdivision.
    """

    NONE = 0
    """
    No subdivision is applied. The mesh remains as is after the initial
    recombination process, potentially leaving some triangles in the mesh.
    """
    ALL_QUADRANGLES = 1
    """
    Subdivides the mesh to convert all elements into quadrangles. This method
    ensures a full-quad mesh by subdividing any remaining triangles after the
    initial recombination process.
    """
    BARYCENTRIC = 3
    """
    Applies barycentric subdivision to the mesh. This method subdivides each
    element by connecting its barycenter to the midpoints of its edges,
    resulting in a refined mesh with increased element count.
    """


class FieldCombination(Enum):
    """
    Controls how cell size fields are combined when they are found at the
    same location.
    """

    MIN = "Min"
    """Use the minimum size."""
    MAX = "Max"
    """Use the maximum size."""
    MEAN = "Mean"
    """Use the mean size."""


class GeneralVerbosity(IntEnum):
    """Level of information printed."""

    SILENT = 0
    """No output is printed. All messages are suppressed."""

    ERRORS = 1
    """
    Only error messages are printed. This level is useful when you want to be
    alerted only to critical issues that prevent correct execution.
    """

    WARNINGS = 2
    """
    Error and warning messages are printed. This level adds important
    cautionary information that doesn't necessarily prevent execution but might
    affect results.
    """

    DIRECT = 3
    """
    Errors, warnings, and direct output from Gmsh commands are printed. This
    level is useful for seeing immediate results of operations without too much
    extra information.
    """

    INFORMATION = 4
    """
    Errors, warnings, direct output, and additional informational messages are
    printed. This level provides more detailed feedback about the progress and
    state of operations.
    """

    STATUS = 5
    """
    Errors, warnings, direct output, information, and status updates are
    printed. This level offers comprehensive feedback, including progress
    indicators for longer operations.
    """

    DEBUG = 99
    """
    All possible output is printed, including detailed debugging information.
    This level is extremely verbose and is typically used for troubleshooting
    or development purposes.
    """
