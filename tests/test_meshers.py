import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg

import pandamesh as pm

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
donut = sg.Polygon(outer, holes=[inner])

other_inner_coords = np.array([(3.0, 4.0), (7.0, 4.0), (7.0, 6.0), (3.0, 6.0)])
other_inner = sg.Polygon(other_inner_coords)
other_hole_coords = np.array(
    [
        (7.0, 3.0),
        (7.0, 4.0),
        (7.0, 6.0),
        (7.0, 7.0),
        (3.0, 7.0),
        (3.0, 6.0),
        (3.0, 4.0),
        (3.0, 3.0),
    ]
)
other_hole = sg.LinearRing(other_hole_coords)
other_donut = sg.Polygon(outer, holes=[other_hole])


def bounds(vertices):
    x, y = vertices.T
    return x.min(), y.min(), x.max(), y.max()


def area(vertices, triangles):
    """
    Compute the area of every triangle in the mesh.
    (Helper for these tests.)
    """
    coords = vertices[triangles]
    u = coords[:, 1] - coords[:, 0]
    v = coords[:, 2] - coords[:, 0]
    return 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def triangle_generate(gdf: gpd.GeoDataFrame, shift: bool):
    mesher = pm.TriangleMesher(gdf, shift_origin=shift)
    return mesher.generate()


def gmsh_generate(gdf: gpd.GeoDataFrame, shift: bool):
    mesher = pm.GmshMesher.get_instance(gdf, shift_origin=shift)
    vertices, faces = mesher.generate()
    return vertices, faces


@pytest.mark.parametrize("generate", [triangle_generate, gmsh_generate])
@pytest.mark.parametrize("shift", [False, True])
def test_empty(generate, shift):
    gdf = gpd.GeoDataFrame(geometry=[line], data={"cellsize": [1.0]})
    with pytest.raises(ValueError, match="No polygons provided"):
        generate(gdf, shift)


@pytest.mark.parametrize("generate", [triangle_generate, gmsh_generate])
@pytest.mark.parametrize("shift", [False, True])
def test_basic(generate, shift):
    polygon = sg.Polygon(outer)
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    gdf["cellsize"] = 1.0
    vertices, triangles = generate(gdf, shift)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, polygon.area)
    assert np.allclose(bounds(vertices), gdf.total_bounds)


@pytest.mark.parametrize("generate", [triangle_generate, gmsh_generate])
@pytest.mark.parametrize("shift", [False, True])
def test_hole(generate, shift):
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    vertices, triangles = generate(gdf, shift)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, donut.area)
    assert np.allclose(bounds(vertices), gdf.total_bounds)


@pytest.mark.parametrize("generate", [triangle_generate, gmsh_generate])
@pytest.mark.parametrize("shift", [False, True])
def test_partial_hole(generate, shift):
    gdf = gpd.GeoDataFrame(geometry=[other_donut, other_inner])
    gdf["cellsize"] = 1.0
    vertices, triangles = generate(gdf, shift)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, other_donut.area + other_inner.area)
    assert np.allclose(bounds(vertices), gdf.total_bounds)


@pytest.mark.parametrize("generate", [triangle_generate, gmsh_generate])
@pytest.mark.parametrize("shift", [False, True])
def test_adjacent_donut(generate, shift):
    inner_coords2 = inner_coords.copy()
    outer_coords2 = outer_coords.copy()
    inner_coords2[:, 0] += 10.0
    outer_coords2[:, 0] += 10.0
    inner2 = sg.LinearRing(inner_coords2)
    outer2 = sg.LinearRing(outer_coords2)
    donut2 = sg.Polygon(outer2, holes=[inner2])

    gdf = gpd.GeoDataFrame(geometry=[donut, donut2])
    gdf["cellsize"] = [1.0, 0.5]
    vertices, triangles = generate(gdf, shift)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)
    assert np.allclose(bounds(vertices), gdf.total_bounds)

    # With a line at y=8.0 and points in the left polygon, at y=2.0
    line1 = sg.LineString([(0.25, 8.0), (9.75, 8.0)])
    line2 = sg.LineString([(10.25, 8.0), (19.75, 8.0)])
    x = np.arange(0.25, 10.0, 0.25)
    y = np.full(x.size, 2.0)
    points = gpd.points_from_xy(x=x, y=y)
    gdf = gpd.GeoDataFrame(geometry=[donut, donut2, line1, line2, *points])
    gdf["cellsize"] = 1.0

    vertices, triangles = generate(gdf, shift)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)
    assert np.allclose(bounds(vertices), gdf.total_bounds)


def test_triangle_properties():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.TriangleMesher(gdf)

    # Should be a float >=0, < 34.0
    with pytest.raises(TypeError):
        mesher.minimum_angle = 10
    with pytest.raises(ValueError):
        mesher.minimum_angle = 35.0

    # Set properties
    mesher.minimum_angle = 10.0
    mesher.conforming_delaunay = False
    mesher.suppress_exact_arithmetic = True
    mesher.maximum_steiner_points = 10
    mesher.delaunay_algorithm = pm.DelaunayAlgorithm.SWEEPLINE
    mesher.consistency_check = True

    # Check whether properties have been set properly
    assert mesher.minimum_angle == 10.0
    assert mesher.conforming_delaunay is False
    assert mesher.suppress_exact_arithmetic is True
    assert mesher.maximum_steiner_points == 10
    assert mesher.delaunay_algorithm == pm.DelaunayAlgorithm.SWEEPLINE
    assert mesher.consistency_check is True

    # Check whether the repr method works
    assert isinstance(mesher.__repr__(), str)

    with pytest.raises(TypeError):
        mesher.conforming_delaunay = "a"

    with pytest.raises(TypeError):
        mesher.suppress_exact_arithmetic = "a"

    with pytest.raises(TypeError):
        mesher.maximum_steiner_points = "a"

    with pytest.raises(ValueError):
        mesher.delaunay_algorithm = "a"

    with pytest.raises(TypeError):
        mesher.consistency_check = "a"


def test_gmsh_properties():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.GmshMesher.get_instance(gdf)

    # Set default values for meshing parameters
    mesher.mesh_algorithm = pm.MeshAlgorithm.FRONTAL_DELAUNAY
    mesher.recombine_all = False
    mesher.mesh_size_extend_from_boundary = False
    mesher.mesh_size_from_points = False
    mesher.mesh_size_from_curvature = True
    mesher.field_combination = pm.FieldCombination.MAX
    mesher.subdivision_algorithm = pm.SubdivisionAlgorithm.BARYCENTRIC

    assert mesher.mesh_algorithm == pm.MeshAlgorithm.FRONTAL_DELAUNAY
    assert mesher.recombine_all is False
    assert mesher.mesh_size_extend_from_boundary is False
    assert mesher.mesh_size_from_points is False
    assert mesher.mesh_size_from_curvature is True
    assert mesher.field_combination == pm.FieldCombination.MAX
    assert mesher.subdivision_algorithm == pm.SubdivisionAlgorithm.BARYCENTRIC

    with pytest.raises(ValueError):
        mesher.mesh_algorithm = "a"

    with pytest.raises(TypeError):
        mesher.recombine_all = "a"

    with pytest.raises(TypeError):
        mesher.mesh_size_extend_from_boundary = "a"

    with pytest.raises(TypeError):
        mesher.mesh_size_from_points = "a"

    with pytest.raises(TypeError):
        mesher.mesh_size_from_curvature = "a"

    with pytest.raises(ValueError):
        mesher.field_combination = "a"

    with pytest.raises(ValueError):
        mesher.subdivision_algorithm = "a"

    with pytest.raises(ValueError):
        mesher.general_verbosity = "a"


def test_gmsh_write(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.GmshMesher.get_instance(gdf)
    path = tmp_path / "a.msh"
    mesher.write(path)
    assert path.exists()


@pytest.mark.parametrize("read_config_files", [True, False])
@pytest.mark.parametrize("interruptible", [True, False])
def test_gmsh_initialization_kwargs(read_config_files, interruptible):
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.GmshMesher.get_instance(
        gdf, read_config_files=read_config_files, interruptible=interruptible
    )
    vertices, triangles = mesher.generate()
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, donut.area)


def test_generate_geodataframe():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0

    result = pm.TriangleMesher(gdf).generate_geodataframe()
    assert isinstance(result, gpd.GeoDataFrame)
    assert np.allclose(result.area.sum(), donut.area)

    mesher = pm.GmshMesher.get_instance(gdf)
    result = mesher.generate_geodataframe()
    assert isinstance(result, gpd.GeoDataFrame)
    assert np.allclose(result.area.sum(), donut.area)

    # Make sure the geodataframe logic can deal with quads as well.
    mesher.recombine_all = True
    result = mesher.generate_geodataframe()
    assert isinstance(result, gpd.GeoDataFrame)
    assert np.allclose(result.area.sum(), donut.area)
