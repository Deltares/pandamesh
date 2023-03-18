import numpy as np

from pandamesh.common import FloatArray, IntArray


def plot(
    vertices: FloatArray,
    faces: IntArray,
    fill_value: int = -1,
    ax=None,
    facecolors: str = "lightgray",
    edgecolors: str = "black",
    **kwargs,
) -> None:
    """
    Plot an unstructured mesh
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots()

    nodes = vertices[faces]
    nodes[faces == fill_value] = np.nan
    collection = PolyCollection(
        nodes, facecolors=facecolors, edgecolors=edgecolors, **kwargs
    )
    ax.add_collection(collection)
    ax.set_aspect(1.0)
    ax.autoscale()
