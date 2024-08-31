from typing import List, NamedTuple, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.geometry as sg

from pandamesh.common import FloatArray, IntArray, coord_dtype, gmsh

Z_DEFAULT = 0.0
POINT_DIM = 0
LINE_DIM = 1
PLANE_DIM = 2


class PolygonInfo(NamedTuple):
    index: int
    size: int
    interior_indices: List[int]
    interior_sizes: List[int]
    polygon_id: int


class LineStringInfo(NamedTuple):
    index: int
    size: int
    embedded_in: Union[int, None]


def polygon_info(
    polygon: sg.Polygon, cellsize: float, index: int, polygon_id: int
) -> Tuple[PolygonInfo, FloatArray, FloatArray, int]:
    exterior_coords = np.array(polygon.exterior.coords)[:-1]
    size = len(exterior_coords)
    vertices = [exterior_coords]
    cellsizes = [np.full(size, cellsize)]
    info = PolygonInfo(index, size, [], [], polygon_id)
    index += size
    for interior in polygon.interiors:
        interior_coords = np.array(interior.coords)[:-1]
        vertices.append(interior_coords)
        size = len(interior_coords)
        cellsizes.append(np.full(size, cellsize))
        info.interior_indices.append(index)
        info.interior_sizes.append(size)
        index += size
    return info, vertices, cellsizes, index


def linestring_info(
    linestring: sg.LineString, cellsize: float, index: int, inside: Union[int, None]
) -> Tuple[LineStringInfo, FloatArray, FloatArray, int]:
    vertices = np.array(linestring.coords)
    size = len(vertices)
    cellsizes = np.full(size, cellsize)
    info = LineStringInfo(index, size, inside)
    index += size
    return info, vertices, cellsizes, index


def add_vertices(vertices, cellsizes, tags) -> None:
    for (x, y), cellsize, tag in zip(vertices, cellsizes, tags):
        gmsh.model.geo.addPoint(x, y, Z_DEFAULT, cellsize, tag)


def add_linestrings(
    features: List[LineStringInfo], tags: IntArray
) -> Tuple[IntArray, IntArray]:
    n_lines = sum(info.size - 1 for info in features)
    line_indices = np.empty(n_lines, dtype=np.int64)
    embedded_in = np.empty(n_lines, dtype=np.int64)
    i = 0
    for info in features:
        point_tags = tags[info.index : info.index + info.size]
        first = point_tags[0]
        for second in point_tags[1:]:
            line_index = gmsh.model.geo.addLine(first, second)
            line_indices[i] = line_index
            embedded_in[i] = info.embedded_in
            first = second
            i += 1
    return line_indices, embedded_in


def add_curve_loop(point_tags: FloatArray) -> int:
    tags = []
    first = point_tags[-1]
    for second in point_tags:
        line_tag = gmsh.model.geo.addLine(first, second)
        tags.append(line_tag)
        first = second
    curve_loop_tag = gmsh.model.geo.addCurveLoop(tags)
    return curve_loop_tag


def add_polygons(
    features: List[PolygonInfo], tags: IntArray
) -> Tuple[List[int], List[int]]:
    plane_tags = []
    for info in features:
        # Add the exterior loop first
        curve_loop_tags = [add_curve_loop(tags[info.index : info.index + info.size])]
        # Now add holes
        for start, size in zip(info.interior_indices, info.interior_sizes):
            loop_tag = add_curve_loop(tags[start : start + size])
            curve_loop_tags.append(loop_tag)
        plane_tag = gmsh.model.geo.addPlaneSurface(curve_loop_tags, tag=info.polygon_id)
        plane_tags.append(plane_tag)
    return curve_loop_tags, plane_tags


def add_points(points: gpd.GeoDataFrame) -> Tuple[IntArray, IntArray]:
    n_points = len(points)
    indices = np.empty(n_points, dtype=np.int64)
    embedded_in = points["__polygon_id"].to_numpy()
    # We have to add points one by one due to the Gmsh addPoint API
    for i, row in enumerate(points.to_dict("records")):
        point = row["geometry"]
        # Rely on the automatic number of gmsh now to generate the indices
        point_index = gmsh.model.geo.addPoint(
            point.x, point.y, Z_DEFAULT, row["cellsize"]
        )
        indices[i] = point_index
    return indices, embedded_in


def collect_polygons(
    polygons: gpd.GeoDataFrame, index: int
) -> Tuple[int, FloatArray, IntArray, List[PolygonInfo]]:
    vertices = []
    cellsizes = []
    features = []
    for row in polygons.to_dict("records"):
        info, coords, cells, index = polygon_info(
            row["geometry"], row["cellsize"], index, row["__polygon_id"]
        )
        vertices.extend(coords)
        cellsizes.extend(cells)
        features.append(info)
    return index, vertices, cellsizes, features


def collect_linestrings(
    linestrings: gpd.GeoDataFrame, index: int
) -> Tuple[int, FloatArray, IntArray, List[LineStringInfo]]:
    vertices = []
    cellsizes = []
    features = []
    for row in linestrings.to_dict("records"):
        info, coords, cells, index = linestring_info(
            row["geometry"], row["cellsize"], index, row["__polygon_id"]
        )
        vertices.append(coords)
        cellsizes.append(cells)
        features.append(info)
    return index, vertices, cellsizes, features


def collect_points(points: gpd.GeoDataFrame) -> FloatArray:
    return np.stack((points["geometry"].x, points["geometry"].y), axis=1)


def embed_where(gdf: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tmp = gpd.sjoin(gdf, polygons, predicate="within", how="inner")
    tmp["cellsize"] = tmp[["cellsize_left", "cellsize_right"]].min(axis=1)
    return tmp[["cellsize", "__polygon_id", "geometry"]]


def add_geometry(
    polygons: gpd.GeoDataFrame, linestrings: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> None:
    if len(polygons) == 0:
        raise ValueError("No polygons provided")
    # Assign unique ids
    polygons["__polygon_id"] = np.arange(1, len(polygons) + 1)

    # Figure out in which polygon the points and linestrings will be embedded.
    linestrings = embed_where(linestrings, polygons)
    embedded_points = embed_where(points, polygons)

    # Collect all coordinates, and store the length and type of every element
    index, poly_vertices, poly_cellsizes, polygon_features = collect_polygons(
        polygons, index=0
    )
    index, line_vertices, line_cellsizes, linestring_features = collect_linestrings(
        linestrings, index
    )
    vertices = np.concatenate(poly_vertices + line_vertices)
    cellsizes = np.concatenate(poly_cellsizes + line_cellsizes)

    # Get the unique vertices, and generate the array of indices pointing to
    # them for every feature
    vertices, indices = np.unique(
        vertices.reshape(-1).view(coord_dtype), return_inverse=True
    )
    vertex_tags = np.arange(1, len(vertices) + 1)
    tags = vertex_tags[indices]
    # Get the smallest cellsize per vertex
    cellsizes = pd.Series(cellsizes).groupby(tags).min().to_numpy()

    # Add all unique vertices. This includes vertices for linestrings and polygons.
    add_vertices(vertices, cellsizes, vertex_tags)
    # Add all geometries to gmsh
    add_polygons(polygon_features, tags)
    linestring_indices, linestring_embedded = add_linestrings(linestring_features, tags)
    gmsh.model.geo.synchronize()

    # Now embed the points and linestrings in the polygons
    for polygon_id, embed_indices in pd.Series(linestring_indices).groupby(
        linestring_embedded
    ):
        gmsh.model.mesh.embed(LINE_DIM, embed_indices, PLANE_DIM, polygon_id)

    if len(embedded_points) > 0:
        point_indices, point_embedded = add_points(embedded_points)
        gmsh.model.geo.synchronize()
        for polygon_id, embed_indices in pd.Series(point_indices).groupby(
            point_embedded
        ):
            gmsh.model.mesh.embed(POINT_DIM, embed_indices, PLANE_DIM, polygon_id)

    gmsh.model.geo.synchronize()
    return


def add_distance_points(points: gpd.GeoSeries) -> IntArray:
    indices = np.empty(len(points), dtype=np.int64)
    for i, (x, y) in enumerate(shapely.get_coordinates(points)):
        indices[i] = gmsh.model.geo.addPoint(x, y, Z_DEFAULT)
    return indices


def add_distance_linestring(
    linestring: sg.LineString,
    distance: float,
) -> IntArray:
    # We could add the line as a Gmsh curve, but Gmsh samples along the line
    # anyway, and the number of points is configured through a discrete
    # sampling value, rather than some spacing distance. So we might as well
    # sample ourselves.
    interpolated = shapely.line_interpolate_point(linestring, distance=distance)
    indices = np.empty(len(interpolated), dtype=np.int64)
    for i, (x, y) in enumerate(interpolated):
        indices[i] = gmsh.model.geo.addPoint(x, y, Z_DEFAULT)
    return indices


def add_distance_linestrings(
    linestrings: gpd.GeoSeries,
    spacing: FloatArray,
) -> IntArray:
    indices = [
        add_distance_linestring(linestring, distance)
        for linestring, distance in zip(linestrings, spacing)
    ]
    return np.concatenate(indices)


def add_distance_polygons(polygons: gpd.GeoSeries, spacing: FloatArray) -> IntArray:
    indices = []
    for polygon, distance in zip(polygons, spacing):
        indices.append(add_distance_linestring(polygon.exterior, distance))
        for interior in polygon.interiors:
            indices.append(add_distance_linestring(interior, distance))
    return np.concatenate(indices)


def add_distance_geometry(geometry: gpd.GeoSeries, spacing: FloatArray) -> IntArray:
    geom_type = geometry.geom_type
    acceptable = ["Polygon", "LineString", "Point"]
    if not geom_type.isin(acceptable).all():
        raise TypeError(
            f"Geometry should be one of {acceptable}. "
            "Call geopandas.GeoDataFrame.explode() to explode multi-part "
            "geometries into multiple single geometries."
        )

    polygons = geometry.loc[geom_type == "Polygon"].copy()
    linestrings = geometry.loc[geom_type == "LineString"].copy()
    points = geometry.loc[geom_type == "Point"].copy()

    indices = []
    if len(points) > 0:
        indices.append(add_distance_points(points))
    if len(linestrings) > 0:
        indices.append(add_distance_linestrings(linestrings, spacing))
    if len(polygons) > 0:
        indices.append(add_distance_polygons(polygons, spacing))
    gmsh.model.geo.synchronize()
    return np.concatenate(indices)
