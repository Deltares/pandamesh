Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

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