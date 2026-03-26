from itertools import product

import jax.numpy as jnp
from compas.utilities import geometric_key, pairwise
from jax_fdm.datastructures import FDMesh

from neural_fdm.generators import evaluate_bezier_surface


def create_mesh_from_tube_generator(generator, config, *args, **kwargs):
    """
    Boundary-supported mesh on a tube. The mesh has group tags.

    Parameters
    ----------
    generator: `TubePointGenerator`
        The tube generator.
    config: `dict`
        The configuration dictionary.

    Returns
    -------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    """
    # shorthands
    tube = generator
    fix_rings = not config["loss"]["shape"]["include"]

    # generate base FD Mesh
    points = tube.points_on_tube()
    points = jnp.reshape(points, (-1, 3))

    num_u = tube.num_levels
    num_v = tube.num_sides
    faces = calculate_mesh_tube_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(points, faces)

    # define structural system
    for vertices in mesh.vertices_on_boundaries():
        mesh.vertices_supports(vertices)

    # tag edges as either rings or cables
    # first assume all edges ar cables
    mesh.edges_attribute("tag", "cable")

    # then, search for ring edges by geometric key
    points = jnp.reshape(points, tube.shape_tube)
    points_rings = points[tube.levels_rings_comp, :, :].tolist()
    gkey_key = mesh.gkey_key()

    num_ring_edges = 0
    for points_ring in points_rings:
        for line in pairwise(points_ring + points_ring[:1]):
            edge = tuple([gkey_key[geometric_key(pt)] for pt in line])
            if not mesh.has_edge(edge):
                u, v = edge
                edge = v, u
            assert mesh.has_edge(edge)
            mesh.edge_attribute(edge, "tag", "ring")
            num_ring_edges += 1

            # NOTE: fix ring supports if no shape error in loss
            if fix_rings:
                mesh.vertices_supports(edge)

    assert num_ring_edges == tube.num_rings * tube.num_sides

    return mesh


def create_mesh_from_bezier_generator(generator, *args, **kwargs):
    """
    Boundary-supported mesh on bezier surface.

    Parameters
    ----------
    generator: `BezierSurfacePointGenerator`
        The bezier surface generator.

    Returns
    -------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    """
    # unpack parameters
    bezier = generator.surface
    u = generator.u
    v = generator.v

    # generate base FD Mesh
    srf_points = bezier.evaluate_points(u, v)
    srf_points = jnp.reshape(srf_points, (-1, 3))

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_mesh_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)

    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())

    return mesh


def create_mesh_from_grid(grid, u, v):
    """
    Boundary-supported mesh on a grid of Bezier control points.

    Parameters
    ----------
    grid: `PointGrid`
        The grid of control points.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].

    Returns
    -------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    """
    # generate base FD Mesh
    srf_points = calculate_bezier_surface_points_from_grid(grid, u, v)

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_mesh_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)

    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())

    return mesh


def calculate_mesh_grid_faces(nx, ny):
    """
    Generate the indices of the mesh faces of the grid.

    Parameters
    ----------
    nx: `int`
        The number of points along the `x` direction.
    ny: `int`
        The number of points along the `y` direction.

    Returns
    -------
    faces: `list` of `list` of `int`
        The indices of the mesh faces.
    """
    faces = []
    for i, j in product(range(nx), range(ny)):
        face = [
            i * (ny + 1) + j,
            (i + 1) * (ny + 1) + j,
            (i + 1) * (ny + 1) + j + 1,
            i * (ny + 1) + j + 1,
        ]
        faces.append(face)

    return faces


def calculate_mesh_tube_faces(nx, ny):
    """
    Generate the indices of the mesh faces of a tube.

    Parameters
    ----------
    nx: `int`
        The number of points along the `x` direction.
    ny: `int`
        The number of points along the `y` direction.

    Returns
    -------
    faces: `list` of `list` of `int`
        The indices of the mesh faces.
    """
    faces = calculate_mesh_grid_faces(nx, ny)

    num_xy = (nx + 1) * (ny + 1)
    starts = range(0, num_xy, ny + 1)
    ends = range(ny, num_xy + ny, ny + 1)

    for (a, b), (d, c) in zip(pairwise(starts), pairwise(ends)):
        face = [d, c, b, a]
        faces.append(face)

    return faces


def calculate_bezier_surface_points_from_grid(grid, u, v):
    """
    Evaluate points on a Bezier surface from a grid of control points.

    Parameters
    ----------
    grid: `PointGrid`
        The grid of control points.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].

    Returns
    -------
    points: `jax.Array`
        The points on the surface.
    """
    # generate control points grid
    control_points = grid.points()

    # sample surface points on bezier
    surface_points = evaluate_bezier_surface(control_points, u, v)

    return jnp.reshape(surface_points, (-1, 3))
