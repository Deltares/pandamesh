Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased
----------

[0.2.0] 2024-09-03
------------------

Fixed
~~~~~

- Previously, :class:`pandamesh.TriangleMesher` would not respect specified
  cell sizes in areas that are fully bounded by linestrings (rather than
  polygons), e.g. three separate lines forming a triangular zone. The reason is
  that Triangle identifies such a zone as a separate region, and the point
  specifying the maximum area is isolated. This has been fixed by checking
  whether linestrings form any coincendental polygons, and including these
  polygons are separate zones.

Added
~~~~~

- :meth:`pandamesh.TriangleMesher.generate_geodataframe()` and -
  :meth:`pandamesh.GmshMesher.generate_geodataframe()` have been added to
  return generated meshes as geodataframes.
- Added :attr:`pandamesh.MeshAlgorithm.QUASI_STRUCTURED_QUAD` as an option.
- Added :class:`pandamesh.Preprocessor` to assist in preparing and cleaning
  geospatial data prior to meshing.
- Added :meth:`pandamesh.GmshMesher.add_threshold_distance_field`,
  :meth:`pandamesh.GmshMesher.add_matheval_distance_field`,
  :meth:`pandamesh.GmshMesher.add_structured_field`, and
  :meth:`pandamesh.GmshMesher.add_structured_field_from_dataarray` to enable
  Gmsh fields from geometry or from raster data.
- Added ``finalize`` keyword to :meth:`pandamesh.GmshMesher.generate` to
  automatically finalize after mesh generation.
- Added :func:`pandamesh.find_edge_intersections` to locate unresolved
  intersection between polygon boundary, linestring, and linearring edges.

Changed
~~~~~~~

- :class:`pandamesh.TriangleMesher` does a cell size to area conversion. This
  previously assumed right-angled triangles. This has been changed to assume
  equilateral triangles instead. This may result in slightly smaller triangles.
- Mesher properties set with :class:`pandamesh.DelaunayAlgorithm`,
  :class:`pandamesh.FieldCombination`, :class:`pandamesh.GeneralVerbosity`,
  :class:`pandamesh.GmshMesher`, :class:`pandamesh.MeshAlgorithm`, or
  :class:`pandamesh.SubdivisionAlgorithm` will now accept one of these enums,
  or the enum member name as a string.
- :class:`pandamesh.TriangleMesher` and :class:`pandamesh.GmshMesher` now take
  a ``shift_origin`` argument to temporarily shift the coordinate system to the
  centroid of the geometries' bounding box to mitigate floating point precision
  problems. This is enabled by default.
- :func:`pandamesh.gmsh_env` now finalizes an existing Gmsh instance prior to
  initializing Gmsh anew.
- :class:`pandamesh.TriangleMesher` and :class:`pandamesh.GmshMesher` will now
  also accept LinearRing geometries (previously only Polygons, LineStrings, and
  Points).
- Added an ``edge_intersection`` keyword to :class:`pandamesh.TriangleMesher`
  and :class:`pandamesh.GmshMesher` to control whether to error, warn, or
  ignore unresolved edge intersections of polygon boundaries, linestrings, and
  linearrings. By default, both meshers will now error if unresolved
  intersections are encountered.

[0.1.6] 2024-07-17
------------------

Added
~~~~~

- :class:`pandamesh.GmshMesher` now takes ``read_config_files`` and ``interruptible``
  as initialization arguments for ``gmsh.``.
  
Fixed
~~~~~

- Compatibility changes for Numpy 2.0.


[0.1.5] 2024-02-06
------------------

Fixed
~~~~~

- Inside of :class:`pandamesh.GmshMesher` a check now occurs before finalization.
  This keeps ``gmsh`` from printing (harmless) errors to the console, which
  previously commonly happened at initialization.
- ``pandamesh`` can now be imported in a sub-thread. ``gmsh`` will not run
  outside of the main interpreter thread, but it previously also prevented 
  the entire import of ``pandamesh``. Attempting to use the
  :class:`pandamesh.GmshMesher` outside of the main thread will result in a
  ``RuntimeError``.

Added
~~~~~

- :class:`pandamesh.GeneralVerbosity` has been added to control the verbosity
  of Gmsh. It can be set via the :attr:`GmshMesher.general_verbosity`
  property. Its default value is ``SILENT``.

Changed
~~~~~~~

- A number of deprecations have been fixed. Most notable is the deprecation
  of ``geopandas.datasets``. The South America geodataframe can now be
  fetched via :func:`pandamesh.data.south_america()`.
- Checking of intersections of linestrings has currently been disabled:
  the current implementation is too strict and resulted in too many false
  positives.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html