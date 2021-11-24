import matplotlib.pyplot as plt
import numpy as np
import pytest

import pandamesh as pm


@pytest.fixture(scope="function")
def mesh():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [1, 4, 2, -1],
        ]
    )
    return vertices, faces


def test_plot(mesh):
    vertices, faces = mesh
    pm.plot(vertices, faces)


def test_plot_optional_args(mesh):
    vertices, faces = mesh

    _, ax = plt.subplots()
    pm.plot(
        vertices,
        faces,
        fill_value=-1,
        ax=ax,
        facecolors="none",
        edgecolors="blue",
        linestyles="dashed",
    )
