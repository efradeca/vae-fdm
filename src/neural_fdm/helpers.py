"""Form-finding helpers: FDM equilibrium computations.

Implements the Force Density Method (FDM) equilibrium equations for
pin-jointed bar systems.

References
----------
[1] Schek, H.J. (1974). The Force Density Method for form finding and
    computation of general networks. CMAME, 3(1):115-134.
[2] Pastrana, R. et al. (2025). Neural FDM. ICLR 2025. Eq. 1, 8.
"""

import jax.numpy as jnp
from jax_fdm.equilibrium import EquilibriumParametersState as FDParametersState
from jax_fdm.equilibrium import EquilibriumState, LoadState, nodes_load_from_faces

# ===============================================================================
# Load helpers
# ===============================================================================

def calculate_area_loads(x, structure, load):
    """
    Convert area loads into vertex loads.

    Parameters
    ----------
    x: `jax.Array`
        The 3D coordinates of the vertices.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    load: `float`
        The vertical load per unit area in the `z` direction.

    Returns
    -------
    vertices_load: `jax.Array`
        The 3D vertex loads.
    """
    x = jnp.reshape(x, (-1, 3))

    # need to convert loads into face loads
    num_faces = structure.num_faces
    faces_load_xy = jnp.zeros(shape=(num_faces, 2))  # (num_faces, xy)
    faces_load_z = jnp.ones(shape=(num_faces, 1)) * load  # (num_faces, xy)
    faces_load = jnp.hstack((faces_load_xy, faces_load_z))

    vertices_load = nodes_load_from_faces(
        x,
        faces_load,
        structure,
        is_local=False
    )

    return vertices_load


def calculate_constant_loads(x, structure, load):
    """
    Create constant vertical vertex loads.

    Parameters
    ----------
    x: `jax.Array`
        The 3D coordinates of the vertices.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    load: `float`
        The vertical load per vertex in the `z` direction.

    Returns
    -------
    vertices_load: `jax.Array`
        The 3D vertex loads.
    """
    num_vertices = structure.num_vertices
    # (num_vertices, xy)
    vertices_load_xy = jnp.zeros(shape=(num_vertices, 2))
    # (num_vertices, xy)
    vertices_load_z = jnp.ones(shape=(num_vertices, 1)) * load

    return jnp.hstack((vertices_load_xy, vertices_load_z))


# ===============================================================================
# Form-finding helpers
# ===============================================================================


def edges_vectors(xyz, connectivity):
    """
    Calculate the unnormalized edge directions (nodal coordinate differences).

    Parameters
    ----------
    xyz: `jax.Array`
        The 3D coordinates of the vertices.
    connectivity: `jax.Array`
        The connectivity matrix of the structure.

    Returns
    -------
    vectors: `jax.Array`
        The edge vectors.
    """
    return connectivity @ xyz


def edges_lengths(vectors):
    """
    Compute the length of the edge vectors.

    Parameters
    ----------
    vectors: `jax.Array`
        The edge vectors.

    Returns
    -------
    lengths: `jax.Array`
        The lengths.
    """
    return jnp.linalg.norm(vectors, axis=1, keepdims=True)


def edges_forces(q, lengths):
    """
    Calculate the force in the edges.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.
    lengths: `jax.Array`
        The edge lengths.

    Returns
    -------
    forces: `jax.Array`
        The forces in the edges.
    """
    return jnp.reshape(q, (-1, 1)) * lengths


def vertices_residuals(q, loads, vectors, connectivity):
    """
    Compute the residual forces on the vertices of the structure.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.
    loads: `jax.Array`
        The loads on the vertices.
    vectors: `jax.Array`
        The edge vectors.
    connectivity: `jax.Array`
        The connectivity matrix of the structure.

    Returns
    -------
    residuals: `jax.Array`
        The residual forces on the vertices.
    """
    return loads - connectivity.T @ (q[:, None] * vectors)


def vertices_residuals_from_xyz(q, loads, xyz, structure):
    """
    Compute the residual forces on the vertices of the structure.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.
    loads: `jax.Array`
        The loads on the vertices.
    xyz: `jax.Array`
        The 3D coordinates of the vertices.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.

    Returns
    -------
    residuals: `jax.Array`
        The residual forces on the vertices.
    """
    connectivity = structure.connectivity

    xyz = jnp.reshape(xyz, (-1, 3))
    vectors = edges_vectors(xyz, connectivity)

    return vertices_residuals(q, loads, vectors, connectivity)


def calculate_equilibrium_state(q, xyz, loads_nodes, structure):
    """
    Assembles an equilibrium state object.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.
    xyz: `jax.Array`
        The 3D coordinates of the vertices.
    loads_nodes: `jax.Array`
        The loads on the vertices.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.

    Returns
    -------
    state: `jax_fdm.EquilibriumState`
        The equilibrium state.
    """
    connectivity = structure.connectivity

    vectors = edges_vectors(xyz, connectivity)
    lengths = edges_lengths(vectors)
    residuals = vertices_residuals(q, loads_nodes, vectors, connectivity)
    forces = edges_forces(q, lengths)

    return EquilibriumState(
        xyz=xyz,
        residuals=residuals,
        lengths=lengths,
        forces=forces,
        loads=loads_nodes,
        vectors=vectors
    )


def calculate_fd_params_state(q, xyz_fixed, loads_nodes):
    """
    Assembles an simulation parameters state.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.
    xyz_fixed: `jax.Array`
        The 3D coordinates of the fixed vertices.
    loads_nodes: `jax.Array`
        The loads on the vertices.

    Returns
    -------
    state: `jax_fdm.EquilibriumParametersState`
        The current state of the simulation parameters.
    """
    return FDParametersState(q, xyz_fixed, LoadState(loads_nodes, 0.0, 0.0))


# =============================================================================
# Reaction forces at supports
# =============================================================================


def compute_reactions(q, loads, xyz, structure):
    """
    Compute reaction forces at support (fixed) nodes.

    In equilibrium, the residual at free nodes is zero. At fixed nodes,
    the residual represents the reaction force the support must provide:

        R_i = sum_j K_ij (x_j - x_i) * q_j - P_i

    This is the negative of the residual at fixed nodes.

    Parameters
    ----------
    q : jax.Array (E,)
        Force densities per edge.
    loads : jax.Array (N, 3)
        Applied loads at all vertices.
    xyz : jax.Array (N, 3)
        Vertex positions.
    structure : EquilibriumStructure
        The mesh structure with connectivity.

    Returns
    -------
    reactions : jax.Array (N_fixed, 3)
        Reaction force vectors at each support node.
    indices_fixed : jax.Array (N_fixed,)
        Indices of the fixed (support) nodes.
    """
    # Compute residuals at ALL nodes (not just free)
    residuals_all = vertices_residuals_from_xyz(q, loads, xyz, structure)

    # Reactions are the negative residuals at fixed nodes
    # At equilibrium, free node residuals are ~0
    # At fixed nodes, residuals represent the imbalance that supports carry
    indices_fixed = structure.indices_fixed
    reactions = -residuals_all[indices_fixed]

    return reactions, indices_fixed


def compute_total_reactions(reactions):
    """
    Compute total reaction forces (sum over all supports).

    Parameters
    ----------
    reactions : jax.Array (N_fixed, 3)
        Reaction force vectors per support.

    Returns
    -------
    total : jax.Array (3,)
        Sum of all reactions [Rx, Ry, Rz].
    """
    return jnp.sum(reactions, axis=0)
