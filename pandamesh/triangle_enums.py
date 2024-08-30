from pandamesh.enum_base import FlexibleEnum


class DelaunayAlgorithm(FlexibleEnum):
    """The type of Delaunay algorithm for Triangle."""

    DIVIDE_AND_CONQUER = ""
    """Default algorithm."""
    INCREMENTAL = "i"
    """
    Uses the incremental algorithm for Delaunay triangulation, rather than
    the divide-and-conquer algorithm.
    """
    SWEEPLINE = "F"
    """
    Uses Steven Fortune's sweepline algorithm for Delaunay triangulation,
    rather than the divide-and-conquer algorithm.
    """
