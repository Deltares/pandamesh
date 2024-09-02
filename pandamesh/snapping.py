from typing import NamedTuple

import numpy as np
import shapely

from pandamesh.common import FloatArray, GeometryArray, IntArray

try:
    from numba import njit  # pragma: no cover
except ImportError:

    def njit(*args, **kwargs):  # pragma: no cover
        # Dummy decorator when numba is not available.
        def decorator(func):
            return func

        return decorator


class MatrixCSR(NamedTuple):
    """
    Compressed Sparse Row matrix. The row indices are compressed; all values
    must therefore be sorted by row number. More or less matches the
    scipy.sparse.csr_matrix.

    NamedTuple for easy ingestion by numba.

    Parameters
    ----------
    data: np.ndarray of floats
        The values of the matrix.
    indices: np.ndarray of integers
        The column numbers of the CSR format.
    indptr: inp.ndarray of integers
        The row index CSR pointer array.
        Values for row i (target index i) are stored in:
        indices[indptr[i]: indptr[i + 1]]
    n: int
        The number of rows.
    m: int
        The number of columns.
    nnz: int
        The number of non-zero values.
    """

    data: FloatArray
    indices: IntArray
    indptr: IntArray
    n: int
    m: int
    nnz: int

    @staticmethod
    def from_triplet(
        row: IntArray, col: IntArray, data: np.ndarray, n: int, m: int
    ) -> "MatrixCSR":
        i = np.cumsum(np.bincount(row, minlength=n))
        indptr = np.empty(i.size + 1, dtype=int)
        indptr[0] = 0
        indptr[1:] = i
        return MatrixCSR(
            data=data,
            indices=col,
            indptr=indptr,
            n=n,
            m=m,
            nnz=data.size,
        )


@njit(inline="always")
def row_slice(A, row: int) -> slice:
    """Return the indices or data slice of a single row."""
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return slice(start, end)


@njit(inline="always")
def columns_and_values(A, slice):
    return zip(A.indices[slice], A.data[slice])


@njit(cache=True)
def _snap_to_nearest(A: MatrixCSR, snap_candidates: IntArray, max_distance) -> IntArray:
    """
    Find a closest target for each node.

    The kD tree distance matrix will have stored for each node the other nodes
    that are within snapping distance. These are the rows in the sparse matrix
    that have more than one entry: the snap_candidates.

    The first condition for a point to become a TARGET is if it hasn't been
    connected to another point yet, i.e. it is UNVISITED. Once a point becomes
    an TARGET, it looks for nearby points within the max_distance. These nearby
    points are connected if: they are UNVISITED (i.e. they don't have a target
    yet), or the current target is closer than the previous.
    """
    UNVISITED = -1
    TARGET = -2
    nearest = np.full(A.n, max_distance + 1.0)
    visited = np.full(A.n, UNVISITED)

    for i in snap_candidates:
        if visited[i] != UNVISITED:
            continue
        visited[i] = TARGET

        # Now iterate through every node j that is within max_distance of node i.
        for j, dist in columns_and_values(A, row_slice(A, i)):
            if i == j or visited[j] == TARGET:
                # Continue if we're looking at the distance to ourselves
                # (i==j), or other node is a target.
                continue
            if visited[j] == UNVISITED or dist < nearest[j]:
                # If unvisited node, or already visited but we're closer, set
                # to i.
                visited[j] = i
                nearest[j] = dist

    return visited


def distance_matrix(geometry: GeometryArray, max_distance: float) -> MatrixCSR:
    n = len(geometry)
    tree = shapely.STRtree(geometry)
    i, j = tree.query(geometry, distance=max_distance, predicate="dwithin")
    coords = shapely.get_coordinates(geometry)
    return MatrixCSR.from_triplet(
        row=i,
        col=j,
        data=np.linalg.norm(coords[i] - coords[j], axis=1),
        n=n,
        m=n,
    )


def snap_nodes(points: GeometryArray, max_distance: float) -> IntArray:
    """
    Snap neigbhoring vertices together that are located within a maximum
    snapping distance from each other.

    If vertices are located within a maximum distance, some of them are snapped
    to their neighbors ("targets"), thereby guaranteeing a minimum distance
    between nodes in the result. The determination of whether a point becomes a
    target itself or gets snapped to another point is primarily based on the
    order in which points are processed and their spatial relationships.

    This function also return an inverse index array. In case of a connectivity
    array, ``inverse`` can be used to index into, yielding the updated
    numbers. E.g.:

    ``updated_face_nodes = inverse[face_nodes]``

    Parameters
    ----------
    xy: nd array of floats of size (N, 2)
    max_distance: float

    Returns
    -------
    index: 1D array of ints of size M
        Which nodes to preserve.
    """
    # First, find all the points that lie within max_distance of each other
    A = distance_matrix(points, max_distance)
    should_snap = np.diff(A.indptr) > 1  # equal to: .getnnz(axis=1) > 1
    if should_snap.any():
        index = np.arange(A.n)
        visited = _snap_to_nearest(
            A=A,
            snap_candidates=index[should_snap],
            max_distance=max_distance,
        )
        targets = visited < 0  # i.e. still UNVISITED or TARGET valued.
        visited[targets] = index[targets]
        deduplicated = np.unique(visited)
        return deduplicated
    else:
        return np.arange(A.n)
