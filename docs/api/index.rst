.. currentmodule:: pandamesh

.. _api:

API Reference
=============

This page provides an auto-generated summary of pandamesh's API.

.. toctree::
   :maxdepth: 1

   changelog

Preprocessing
-------------

.. autosummary::
   :toctree: api/

    Preprocessor
    Preprocessor.unify_polygons
    Preprocessor.merge_polygons
    Preprocessor.clip_lines
    Preprocessor.unify_lines
    Preprocessor.interpolate_lines_to_points
    Preprocessor.snap_points
    Preprocessor.clip_points
    Preprocessor.to_geodataframe
    find_edge_intersections
    find_proximate_polygon_points

Triangle
--------

.. autosummary::
   :toctree: api/

    TriangleMesher
    TriangleMesher.generate
    TriangleMesher.generate_geodataframe
    TriangleMesher.generate_ugrid
    TriangleMesher.minimum_angle
    TriangleMesher.conforming_delaunay
    TriangleMesher.suppress_exact_arithmetic
    TriangleMesher.maximum_steiner_points
    TriangleMesher.delaunay_algorithm
    TriangleMesher.consistency_check

Triangle Enumerators
--------------------

.. autosummary::
   :toctree: api/
   :template: enums.rst

    DelaunayAlgorithm

Gmsh
----

.. autosummary::
   :toctree: api/

    GmshMesher
    GmshMesher.generate
    GmshMesher.generate_geodataframe
    GmshMesher.generate_ugrid
    GmshMesher.mesh_algorithm
    GmshMesher.recombine_all
    GmshMesher.mesh_size_extend_from_boundary
    GmshMesher.mesh_size_from_points
    GmshMesher.mesh_size_from_curvature
    GmshMesher.field_combination
    GmshMesher.subdivision_algorithm
    GmshMesher.general_verbosity
    GmshMesher.add_matheval_distance_field
    GmshMesher.add_threshold_distance_field
    GmshMesher.add_structured_field
    GmshMesher.add_structured_field_from_dataarray
    GmshMesher.fields
    GmshMesher.clear_fields
    GmshMesher.write
    GmshMesher.finalize
    GmshMesher.finalize_gmsh
    gmsh_env

Gmsh Enumerators
----------------

.. autosummary::
   :toctree: api/
   :template: enums.rst

    FieldCombination
    GeneralVerbosity
    GmshMesher
    MeshAlgorithm
    SubdivisionAlgorithm
