"""Graph data structures for GNN-based form-finding."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# ===============================================================================
# Graph data container
# ===============================================================================

class GraphData(NamedTuple):
    """Lightweight graph container for a single mesh graph.

    Parameters
    ----------
    node_features : Array
        Vertex positions of shape ``(N, 3)``.
    edge_index : Array
        Sender/receiver indices in COO format of shape ``(2, E)``.
    num_nodes : int
        Number of nodes (vertices) in the graph.
    num_edges : int
        Number of directed edges in the graph.
    """
    node_features: Float[Array, "N 3"]
    edge_index: Int[Array, "2 E"]
    num_nodes: int
    num_edges: int


# ===============================================================================
# Construction helpers
# ===============================================================================

def structure_to_graph(
    structure,
    xyz_flat: Float[Array, "N3"],
) -> GraphData:
    """Convert an EquilibriumMeshStructure and flat xyz array to a GraphData.

    Parameters
    ----------
    structure : EquilibriumMeshStructure
        The equilibrium structure (carries topology information).
    xyz_flat : Array
        Flat vertex positions of length ``num_vertices * 3``.

    Returns
    -------
    GraphData
        A graph whose nodes are the mesh vertices and whose edges follow the
        mesh connectivity.
    """
    node_features = jnp.reshape(xyz_flat, (-1, 3))
    num_nodes = int(node_features.shape[0])

    # The structure stores a connectivity matrix; extract the edge list from it.
    # ``structure.connectivity`` is a (num_edges, num_nodes) matrix where each
    # row has exactly one +1 (sender) and one -1 (receiver).
    connectivity = jnp.array(structure.connectivity)
    num_edges = int(connectivity.shape[0])

    senders = jnp.argmax(connectivity, axis=1)       # +1 entries
    receivers = jnp.argmax(-connectivity, axis=1)     # -1 entries

    edge_index = jnp.stack([senders, receivers], axis=0)  # (2, E)

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


def edge_index_from_mesh(mesh) -> Int[Array, "2 E"]:
    """Extract a COO edge index from an FDMesh.

    Iterates ``mesh.edges()`` to collect ``(u, v)`` pairs and returns a JAX
    array of shape ``(2, num_edges)``.

    Parameters
    ----------
    mesh : FDMesh
        A COMPAS / jax_fdm mesh datastructure.

    Returns
    -------
    edge_index : Array
        Integer array of shape ``(2, num_edges)``.
    """
    senders = []
    receivers = []
    for u, v in mesh.edges():
        senders.append(u)
        receivers.append(v)

    return jnp.array([senders, receivers], dtype=jnp.int32)


# ===============================================================================
# Edge feature computation
# ===============================================================================

def compute_edge_features(
    node_features: Float[Array, "N 3"],
    edge_index: Int[Array, "2 E"],
) -> tuple[Float[Array, "E 3"], Float[Array, "E 1"]]:
    """Compute edge features from node positions and edge connectivity.

    For each edge ``(i, j)`` the features are:

    * **relative_pos** -- ``node_features[j] - node_features[i]``  (shape ``(E, 3)``)
    * **distance** -- Euclidean length of the relative position vector  (shape ``(E, 1)``)

    Parameters
    ----------
    node_features : Array
        Vertex positions of shape ``(N, 3)``.
    edge_index : Array
        COO edge index of shape ``(2, E)``.

    Returns
    -------
    relative_positions : Array
        Relative position vectors of shape ``(E, 3)``.
    distances : Array
        Euclidean distances of shape ``(E, 1)``.
    """
    senders = edge_index[0]    # (E,)
    receivers = edge_index[1]  # (E,)

    relative_positions = node_features[receivers] - node_features[senders]  # (E, 3)
    distances = jnp.linalg.norm(relative_positions, axis=-1, keepdims=True)  # (E, 1)

    return relative_positions, distances
