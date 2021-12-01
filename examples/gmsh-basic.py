"""
Basic Gmsh Example
======================

In this example we'll create some basic geometries and turn them into meshes.
to illustrate some of the mesh generation features that Gmsh provides in
combination with polygon, point, and linestring geometries represented by
geopandas.

The :py:class:`GmshMesher` supports the geometry show in the basic Triangle
example. Not all of those are repeated here, rather we focus on some of the
additional features that Gmsh provides.
"""
# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry as sg

import pandamesh as pm

# %%
# A simple rectangular mesh
# -------------------------
#
# The most simple example is perhaps a rectangle. We'll create a vector
# geometry, store this in a geodataframe, and associate a cell size.

polygon = sg.Polygon(
    [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ]
)
gdf = gpd.GeoDataFrame(geometry=[polygon])
gdf["cellsize"] = 2.0

# %%
# We'll use this polygon to generate a mesh. We start by initializing a
# TriangleMesher, which is a simple wrapper around the Python bindings to the
# Gmsh C++-library. This wrapper extracts the coordinates and presents them
# in the appropriate manner for Gmsh.

mesher = pm.GmshMesher(gdf)
vertices, triangles = mesher.generate()
pm.plot(vertices, triangles)

# %%
# As the name suggests, Triangle only generates triangular meshes. Gmsh is
# capable of generating quadrilateral-dominant meshes, and has a lot more bells
# and whistles for defining cellsizes.

line = sg.LineString([(2.0, 8.0), (8.0, 2.0)])
gdf = gpd.GeoDataFrame(geometry=[polygon, line])
gdf["cellsize"] = [2.0, 0.2]

fig, (ax0, ax1) = plt.subplots(ncols=2)

mesher = pm.TriangleMesher(gdf)
vertices, triangles = mesher.generate()
pm.plot(vertices, triangles, ax=ax0)

mesher = pm.GmshMesher(gdf)
vertices, triangles = mesher.generate()
pm.plot(vertices, triangles, ax=ax1)

# %%
# Gmsh allows for specifying cell sizes not just on polygons (regions) like
# Triangle (left), but on individual vertices as well, as is visible around the
# diagonal (right).
#
# Defaults
# --------
#
# The GmshMesher class is initialized with a number of default parameters:

print(mesher)

# %%
# Quadrilateral meshes
# --------------------
#
# It is also capable of generating quadrilateral (dominant) meshes:

gdf = gpd.GeoDataFrame(geometry=[polygon])
gdf["cellsize"] = 2.0
mesher = pm.GmshMesher(gdf)
mesher.recombine_all = True
vertices, faces = mesher.generate()

pm.plot(vertices, faces)

# %%
# Writing to file
# ---------------
# It's also possible to use the Python bindings to write a Gmsh ``.msh`` file.
# This file can be opened using the Gmsh GUI to e.g. inspect the generated
# mesh.

mesher.write("my-mesh.msh")
# %%
