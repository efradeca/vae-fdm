"""Tests for the GNN encoder and graph utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from neural_fdm.gnn import (
    GNNEncoder,
    MessagePassingLayer,
)
from neural_fdm.graph import (
    GraphData,
    compute_edge_features,
    edge_index_from_mesh,
    structure_to_graph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_edge_index(small_mesh):
    """COO edge index extracted from the small 4-node mesh."""
    return edge_index_from_mesh(small_mesh)


@pytest.fixture
def small_num_edges(small_mesh):
    """Number of edges in the small mesh."""
    return small_mesh.number_of_edges()


@pytest.fixture
def small_edges_signs(small_num_edges):
    """All-compression sign vector for the small mesh."""
    return -jnp.ones(small_num_edges)


@pytest.fixture
def gnn_encoder(small_edges_signs, small_edge_index, prng_key):
    """A small GNNEncoder built from the small mesh fixtures."""
    return GNNEncoder(
        edges_signs=small_edges_signs,
        q_shift=0.0,
        hidden_dim=16,
        num_layers=2,
        edge_index=small_edge_index,
        key=prng_key,
    )


# ---------------------------------------------------------------------------
# Graph data tests
# ---------------------------------------------------------------------------

class TestGraphData:

    def test_graph_data_from_mesh(self, small_structure, small_xyz):
        """structure_to_graph produces a valid GraphData."""
        graph = structure_to_graph(small_structure, small_xyz)
        assert isinstance(graph, GraphData)
        assert graph.node_features.shape == (graph.num_nodes, 3)
        assert graph.edge_index.shape[0] == 2

    def test_edge_index_shape(self, small_mesh, small_edge_index):
        """edge_index should be (2, num_edges)."""
        num_edges = small_mesh.number_of_edges()
        assert small_edge_index.shape == (2, num_edges)

    def test_edge_index_values(self, small_mesh, small_edge_index):
        """All indices in edge_index must be valid vertex indices."""
        num_vertices = small_mesh.number_of_vertices()
        assert jnp.all(small_edge_index >= 0)
        assert jnp.all(small_edge_index < num_vertices)

    def test_edge_features_shape(self, small_xyz, small_edge_index):
        """Relative positions are (E, 3) and distances are (E, 1)."""
        node_features = jnp.reshape(small_xyz, (-1, 3))
        rel_pos, dists = compute_edge_features(node_features, small_edge_index)
        num_edges = small_edge_index.shape[1]
        assert rel_pos.shape == (num_edges, 3)
        assert dists.shape == (num_edges, 1)

    def test_edge_features_distances_positive(self, small_xyz, small_edge_index):
        """Edge distances must be non-negative."""
        node_features = jnp.reshape(small_xyz, (-1, 3))
        _, dists = compute_edge_features(node_features, small_edge_index)
        assert jnp.all(dists >= 0.0)


# ---------------------------------------------------------------------------
# MessagePassingLayer tests
# ---------------------------------------------------------------------------

class TestMessagePassingLayer:

    def test_output_shape(self, small_edge_index, prng_key):
        """Output should have the same shape as input node features."""
        node_dim = 16
        edge_dim = 4
        num_nodes = 4

        layer = MessagePassingLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=16,
            key=prng_key,
        )

        node_feat = jnp.ones((num_nodes, node_dim))
        edge_feat = jnp.ones((small_edge_index.shape[1], edge_dim))

        out = layer(node_feat, small_edge_index, edge_feat)
        assert out.shape == (num_nodes, node_dim)


# ---------------------------------------------------------------------------
# GNNEncoder tests
# ---------------------------------------------------------------------------

class TestGNNEncoder:

    def test_output_shape(self, gnn_encoder, small_xyz, small_num_edges):
        """Encoder output should have shape (num_edges,)."""
        q = gnn_encoder(small_xyz)
        assert q.shape == (small_num_edges,)

    def test_sign_constraint(self, gnn_encoder, small_xyz):
        """Output signs should match edges_signs (negative for compression)."""
        q = gnn_encoder(small_xyz)
        edges_signs = gnn_encoder.edges_signs

        signs_match = jnp.sign(q) == edges_signs
        is_zero = q == 0.0
        assert jnp.all(signs_match | is_zero)

    def test_deterministic(self, gnn_encoder, small_xyz):
        """Same input should produce the same output."""
        q1 = gnn_encoder(small_xyz)
        q2 = gnn_encoder(small_xyz)
        assert jnp.allclose(q1, q2)

    def test_jit(self, gnn_encoder, small_xyz, small_num_edges):
        """GNNEncoder can be JIT compiled."""
        jit_fn = eqx.filter_jit(gnn_encoder)
        q = jit_fn(small_xyz)
        assert q.shape == (small_num_edges,)

    def test_grad(self, gnn_encoder, small_xyz):
        """Gradients can be computed through the GNN encoder."""

        @eqx.filter_grad
        def grad_fn(encoder, x):
            q = encoder(x)
            return jnp.sum(q)

        grads = grad_fn(gnn_encoder, small_xyz)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
        assert has_nonzero

    def test_different_xyz(self, gnn_encoder, small_xyz):
        """Different input xyz should produce different q."""
        q1 = gnn_encoder(small_xyz)
        xyz2 = small_xyz + 0.5
        q2 = gnn_encoder(xyz2)
        assert not jnp.allclose(q1, q2)

    def test_explicit_edge_index(self, gnn_encoder, small_xyz, small_edge_index, small_num_edges):
        """Passing edge_index explicitly should work and match stored."""
        q_stored = gnn_encoder(small_xyz)
        q_explicit = gnn_encoder(small_xyz, edge_index=small_edge_index)
        assert jnp.allclose(q_stored, q_explicit)
        assert q_explicit.shape == (small_num_edges,)

    def test_q_shift(self, small_edges_signs, small_edge_index, small_xyz, prng_key):
        """Non-zero q_shift increases the magnitude of force densities."""
        encoder_no_shift = GNNEncoder(
            edges_signs=small_edges_signs,
            q_shift=0.0,
            hidden_dim=16,
            num_layers=2,
            edge_index=small_edge_index,
            key=prng_key,
        )
        encoder_with_shift = GNNEncoder(
            edges_signs=small_edges_signs,
            q_shift=1.0,
            hidden_dim=16,
            num_layers=2,
            edge_index=small_edge_index,
            key=prng_key,
        )

        q_no = encoder_no_shift(small_xyz)
        q_yes = encoder_with_shift(small_xyz)

        # With shift, magnitudes should be larger
        assert jnp.all(jnp.abs(q_yes) >= jnp.abs(q_no) - 1e-6)
