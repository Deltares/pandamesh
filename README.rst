pandamesh
=========

.. image:: https://img.shields.io/github/workflow/status/deltares/pandamesh/ci?style=flat-square
   :target: https://github.com/deltares/pandamesh/actions?query=workflows%3Aci
.. image:: https://img.shields.io/codecov/c/github/deltares/pandamesh.svg?style=flat-square
   :target: https://app.codecov.io/gh/deltares/pandamesh
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
   :target: https://github.com/psf/black

This package translates geospatial vector data (points, lines, or polygons) to
unstructured meshes.

.. code:: python

   import geopandas as gpd
   import pandamesh as pm

   # Get some sample data from geopandas.
   world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

   # Select South America, explode any multi-polygon, and project it to UTM20.
   south_america = world[world["continent"] == 'South America']
   south_america = south_america.explode().reset_index().to_crs(epsg=32620)

   # Set a maximum cell size of 500 km and generate a mesh.
   south_america["cellsize"] = 500_000.0
   mesher = pm.TriangleMesher(south_america)
   vertices, faces = mesher.generate()
   
.. image:: https://raw.githubusercontent.com/Deltares/pandamesh/main/docs/_static/pandamesh-demo.png
  :target: https://github.com/deltares/xugrid

The package converts geospatial data, presented as
`geopandas`_ `GeoDataFrames`_, to unstructured meshes using the open source
high quality mesh generators:

* Christophe Geuzaine and Jean-François Remacle's `Gmsh`_
* Jonathan Shewchuk's `Triangle`_

utilizing the respective Python API's, available at:

* https://pypi.org/project/gmsh/
* https://pypi.org/project/triangle/
  
For completeness, the source code of both projects can be found at:

* https://gitlab.onelab.info/gmsh/gmsh, under ``api/gmsh.py``
* https://github.com/drufat/triangle

These APIs are wrapped in two lightweight classes: ``pandamesh.TriangleMesher``
and ``pandamesh.GmshMesher``. Both are initialized with a GeoDataFrame defining
the geometry features of the mesh. During initialization, geometries are
checked for overlaps and intersections, as the mesh generators cannot deal with
these.  Generated meshes are returned as two numpy arrays: the coordinates of
the vertices, and the connectivity of the mesh faces to these vertices (as is
`usual`_ for many unstructured grid representations).

GeoPandas is not suited for geometries that "wrap around" the world.
Consequently, this package cannot generate meshes for e.g. a sphere.

Installation
------------

.. code:: console

    pip install pandamesh
    
Documentation
-------------

.. image:: https://img.shields.io/github/workflow/status/deltares/pandamesh/docs?style=flat-square
   :target: https://deltares.github.io/pandamesh/

.. _geopandas: https://geopandas.org/
.. _GeoDataFrames: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html
.. _Gmsh: https://gmsh.info/
.. _Triangle: https://www.cs.cmu.edu/~quake/triangle.html
.. _usual: https://ugrid-conventions.github.io/ugrid-conventions/
