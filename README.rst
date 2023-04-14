pandamesh
=========

.. image:: https://img.shields.io/github/actions/workflow/status/deltares/pandamesh/ci.yml?style=flat-square
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
  :target: https://github.com/deltares/pandamesh

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

.. image:: https://img.shields.io/github/actions/workflow/status/deltares/pandamesh/docs.yml?style=flat-square
   :target: https://deltares.github.io/pandamesh/
   
Find the documentation here: deltares.github.io/pandamesh/
   
Other projects
--------------

Pandamesh has been developed because none of the existing packages provide a
straightforward scripting based approach to converting 2D vector geometries to
2D unstructured grids.

Examples of other packages which work with unstructured meshes are listed below.

See also `this list`_ for many other mesh generation tools.

pygmsh
******

The `pygmsh Python package`_  provides useful abstractions from Gmsh's own
Python interface so you can create complex geometries more easily. It also
provides tools for 3D operations (e.g. extrusions).

qgis-gsmh
*********

qgis-gmsh generates geometry input files for the GMSH mesh generator and
converts the Gmsh mesh files to shapefiles that can be imported into QGIS.

* Lambrechts, J., Comblen, R., Legat, V., Geuzaine, C., & Remacle, J. F. (2008).
  Multiscale mesh generation on the sphere. Ocean Dynamics, 58(5-6), 461-473.
* Remacle, J. F., & Lambrechts, J. (2018). Fast and robust mesh generation on
  the sphere—Application to coastal domains. Computer-Aided Design, 103, 14-23.
  https://doi.org/10.1016/j.cad.2018.03.002  

Source: https://github.com/ccorail/qgis-gmsh

Shingle
*******

Shingle provides generalised self-consistent and automated domain
discretisation for multi-scale geophysical models.

* Candy, A. S., & Pietrzak, J. D. (2018). Shingle 2.0: generalising
  self-consistent and automated domain discretisation for multi-scale
  geophysical models. Geoscientific Model Development, 11(1), 213-234.
  https://doi.org/10.5194/gmd-11-213-2018

Source: https://github.com/shingleproject/Shingle 

Website: http://shingleproject.org/index_shingle1.0.html

.. _geopandas: https://geopandas.org/
.. _GeoDataFrames: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html
.. _Gmsh: https://gmsh.info/
.. _Triangle: https://www.cs.cmu.edu/~quake/triangle.html
.. _usual: https://ugrid-conventions.github.io/ugrid-conventions/
.. _pygmsh Python package: https://github.com/nschloe/pygmsh
.. _this list: https://github.com/nschloe/awesome-scientific-computing#meshing
