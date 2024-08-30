from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from pandamesh.common import FloatArray, Grouper, IntArray, flatten, flatten_geometry


def segmentize_linestrings(linestrings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if len(linestrings) == 0:
        return linestrings
    condition = linestrings["cellsize"].notnull()
    segmentized = linestrings.loc[condition].copy()
    segmentized["geometry"] = segmentized.segmentize(
        max_segment_length=segmentized["cellsize"]
    )
    return pd.concat([linestrings.loc[~condition], segmentized])


def add_linestrings(linestrings: gpd.GeoSeries) -> Tuple[FloatArray, IntArray]:
    if len(linestrings) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    geometry = linestrings.geometry.to_numpy()
    n_vertex = shapely.get_num_coordinates(geometry)
    vertices, index = shapely.get_coordinates(geometry, return_index=True)

    vertex_numbers = np.arange(n_vertex.sum())
    segments = np.empty((n_vertex.sum() - 1, 2), dtype=np.int32)
    segments[:, 0] = vertex_numbers[:-1]
    segments[:, 1] = vertex_numbers[1:]
    keep = np.diff(index) == 0
    segments = segments[keep]

    return vertices, segments


def add_polygons(
    polygons: gpd.GeoDataFrame,
) -> Tuple[FloatArray, IntArray, FloatArray]:
    is_region = polygons["cellsize"].notnull()
    n_region = is_region.sum()
    regions = np.empty((n_region, 4), dtype=np.float64)
    region_points = polygons[is_region].representative_point()
    regions[:, 0] = region_points.x
    regions[:, 1] = region_points.y
    regions[:, 2] = np.arange(n_region)
    cellsize = polygons[is_region]["cellsize"].to_numpy()
    # Assume equilateral triangles for cell size to area conversion.
    regions[:, 3] = 0.25 * np.sqrt(3) * cellsize * cellsize

    boundary = polygons.boundary.explode(index_parts=True).geometry
    vertices, segments = add_linestrings(boundary)
    return vertices, segments, regions


def add_points(points: gpd.GeoDataFrame) -> FloatArray:
    vertices = np.empty((len(points), 2), dtype=np.float64)
    if len(points) > 0:
        vertices[:, 0] = points.geometry.x
        vertices[:, 1] = points.geometry.y
    return vertices


def polygon_holes(
    polygons: gpd.GeoDataFrame,
):
    """
    Return a point for every hole in every polygon.

    Triangle recognizes holes as a point contained by segments.
    """
    # An interior may be a true hole, or it could be (partially!) filled with
    # another polygon. Find out if this is the case: get the interiors, and
    # diff them with any polygon inside.
    inner_rings = flatten(polygons.interiors)
    interiors = np.asarray(flatten_geometry(shapely.polygonize(inner_rings)))
    tree = shapely.STRtree(interiors)
    index_inside, index_interior = tree.query(
        polygons.representative_point(), predicate="within"
    )
    nothing_inside = np.full(len(interiors), True)
    nothing_inside[index_interior] = False

    points = [gpd.GeoSeries(interiors[nothing_inside]).representative_point()]
    if len(index_inside) > 0:
        for interior, polygons_inside in Grouper(
            a=interiors,
            a_index=index_interior,
            b=polygons.geometry,
            b_index=index_inside,
        ):
            all_polygons = shapely.unary_union(polygons_inside)
            true_holes = shapely.difference(interior, all_polygons)
            if shapely.is_empty(true_holes).all():
                continue
            hole_points = gpd.GeoSeries(true_holes).explode().representative_point()
            points.append(hole_points)

    points = gpd.GeoSeries(np.concatenate(points))
    if len(points) > 0:
        return add_points(points)
    else:
        return None


def _polygon_polygon_difference(
    a: gpd.GeoDataFrame, b: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out = a.copy()
    out["geometry"] = out["geometry"].difference(b["geometry"].union_all())
    return out.loc[out.area > 0].copy()


def convert_linestring_rings(polygons: gpd.GeoDataFrame, linestrings: gpd.GeoDataFrame):
    # Check if linestrings contain any rings. Triangle will treat such a ring
    # as a region on its own.
    linestring_polygons = linestrings.polygonize()
    if linestring_polygons.empty:
        # No rings found
        return polygons

    # Assign the cell size to the created polygons Do a cheap check: see
    # whether a vertex falls within the polygons.
    polygons_inside = polygons.sjoin(
        gpd.GeoDataFrame(
            geometry=shapely.get_point(linestring_polygons.exterior, index=0)
        ),
        predicate="contains",
        how="right",
    )
    # Set the original geometry back.
    polygons_inside["geometry"] = linestring_polygons
    # Any linestring polygon outside will have a cellsize of NaN; remove those
    # entries.
    polygons_inside = polygons_inside.loc[polygons_inside["cellsize"].notnull()]
    # Ensure we burn the polygon holes into the newly created linestrings
    # polygons.
    new_polygons = polygons_inside.loc[:, ["cellsize", "geometry"]].copy()
    new_polygons["geometry"] = new_polygons["geometry"].intersection(
        polygons["geometry"].union_all()
    )

    # Make room for the new polygons
    diffed_polygons = _polygon_polygon_difference(
        a=polygons,
        b=polygons_inside,
    )
    return pd.concat((diffed_polygons, new_polygons))


def unique_vertices_and_segments(vertices, segments):
    # If the geometry is closed (LinearRings), the final vertex of every
    # feature is discarded, since the repeats will segfault Triangle.
    vertices, inverse = np.unique(vertices, return_inverse=True, axis=0)
    inverse = inverse.ravel()
    segments = inverse[segments]
    # Now remove duplicated segments.
    segments = np.unique(segments, axis=0)
    return vertices, segments


def collect_geometry(
    polygons: gpd.GeoDataFrame, linestrings: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> Tuple[FloatArray, IntArray, FloatArray]:
    if len(polygons) == 0:
        raise ValueError("No polygons provided")
    linestrings = segmentize_linestrings(linestrings)
    polygons = convert_linestring_rings(polygons, linestrings)
    polygon_vertices, polygon_segments, regions = add_polygons(polygons)
    linestring_vertices, linestring_segments = add_linestrings(linestrings.geometry)
    point_vertices = add_points(points)
    linestring_segments += polygon_vertices.shape[0]
    vertices = np.concatenate([polygon_vertices, linestring_vertices, point_vertices])
    segments = np.concatenate([polygon_segments, linestring_segments])
    vertices, segments = unique_vertices_and_segments(vertices, segments)
    return vertices, segments, regions
