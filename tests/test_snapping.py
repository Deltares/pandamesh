import os

import numpy as np
import pytest
import shapely

from pandamesh.snapping import MatrixCSR, columns_and_values, row_slice, snap_nodes


def numba_enabled() -> bool:
    return os.environ.get("NUMBA_DISABLE_JIT") != "1"


@pytest.fixture(scope="function")
def csr_matrix():
    i = np.repeat(np.arange(5), 2)
    j = np.arange(10)
    v = np.full(10, 0.5)
    return MatrixCSR.from_triplet(i, j, v, 5, 10)


@pytest.mark.skipif(
    numba_enabled(),
    reason="Function returns a slice object; python and no-python slices don't mix.",
)
def test_row_slice(csr_matrix):
    # These functions work fine if called inside of other numba functions when
    # numba is enabled.
    assert row_slice(csr_matrix, 0) == slice(0, 2, None)


@pytest.mark.skipif(
    numba_enabled(),
    reason="Function returns a zip object; python and no-python zips don't mix.",
)
def test_columns_and_values(csr_matrix):
    # These functions work fine if called inside of other numba functions when
    # numba is enabled.
    zipped = columns_and_values(csr_matrix, row_slice(csr_matrix, 0))
    result = list(zipped)
    assert result == [(0, 0.5), (1, 0.5)]


def test_snap__three_points_horizontal():
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros_like(x)
    xy = shapely.points(x, y)
    index = snap_nodes(xy, 0.1)
    assert np.array_equal(index, [0, 1, 2])

    index = snap_nodes(xy, 1.0)
    assert np.array_equal(index, [0, 2])

    index = snap_nodes(xy, 2.0)
    assert np.array_equal(index, [0])


def test_snap__three_points_diagonal():
    x = y = np.array([0.0, 1.0, 1.5])
    xy = shapely.points(x, y)
    index = snap_nodes(xy, 0.1)
    assert np.array_equal(index, [0, 1, 2])

    # hypot(0.5, 0.5) = 0.707...
    index = snap_nodes(xy, 0.71)
    assert np.array_equal(index, [0, 1])

    # hypot(1, 1) = 1.414...
    index = snap_nodes(xy, 1.42)
    assert np.array_equal(index, [0, 2])
