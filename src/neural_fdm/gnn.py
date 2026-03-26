"""Graph Neural Network encoder for topology-agnostic form-finding.

Implements a message-passing neural network (MPNN) that operates directly on
the mesh graph. The encoder takes vertex positions and mesh connectivity as
input and produces per-edge force density values, matching the interface of
:class:`neural_fdm.models.Encoder`.

The message-passing architecture follows the framework of Gilmer et al. (2017)
and Battaglia et al. (2018), adapted for structural force density prediction.

References
----------
[1] Gilmer, J. et al. (2017). Neural Message Passing for Quantum Chemistry.
    ICML 2017. arXiv:1704.01212
[2] Battaglia, P. et al. (2018). Relational inductive biases, deep learning,
    and graph networks. arXiv:1806.01261
[3] Pastrana, R. et al. (2025). Neural FDM. ICLR 2025. Section 6.1
    mentions graph networks as future work for the encoder.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from neural_fdm.graph import compute_edge_features
from neural_fdm.models import Encoder

# ===============================================================================
# Message-passing layer
# ===============================================================================

class MessagePassingLayer(eqx.Module):
    """Single message-passing layer with edge and node updates.

    For each edge ``(i, j)`` with edge features ``e_ij``:

    1. **Message**:   ``m_ij = MLP_msg([h_i, h_j, e_ij])``
    2. **Aggregate**: ``agg_i = sum_{j in N(i)} m_ij``
    3. **Update**:    ``h_i' = MLP_upd([h_i, agg_i])``

    Parameters
    ----------
    message_mlp : eqx.nn.MLP
        MLP that computes per-edge messages.
    update_mlp : eqx.nn.MLP
        MLP that updates node embeddings from aggregated messages.
    """

    message_mlp: eqx.nn.MLP
    update_mlp: eqx.nn.MLP

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)

        # Message MLP: [h_sender, h_receiver, e_ij] -> hidden_dim
        self.message_mlp = eqx.nn.MLP(
            in_size=2 * node_dim + edge_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jax.nn.elu,
            key=k1,
        )

        # Update MLP: [h_i, agg_i] -> node_dim
        self.update_mlp = eqx.nn.MLP(
            in_size=node_dim + hidden_dim,
            out_size=node_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jax.nn.elu,
            key=k2,
        )

    def __call__(
        self,
        node_features: Float[Array, "N node_dim"],
        edge_index: Int[Array, "2 E"],
        edge_features: Float[Array, "E edge_dim"],
    ) -> Float[Array, "N node_dim"]:
        """Run one round of message passing.

        Parameters
        ----------
        node_features : Array
            Node embeddings of shape ``(N, node_dim)``.
        edge_index : Array
            COO edge indices ``[senders, receivers]`` of shape ``(2, E)``.
        edge_features : Array
            Edge features of shape ``(E, edge_dim)``.

        Returns
        -------
        Array
            Updated node embeddings of shape ``(N, node_dim)``.
        """
        senders = edge_index[0]    # (E,)
        receivers = edge_index[1]  # (E,)

        # Gather sender and receiver features
        h_senders = node_features[senders]      # (E, node_dim)
        h_receivers = node_features[receivers]   # (E, node_dim)

        # Compute messages
        msg_input = jnp.concatenate([h_senders, h_receivers, edge_features], axis=-1)
        messages = jax.vmap(self.message_mlp)(msg_input)  # (E, hidden_dim)

        # Aggregate messages to receiver nodes via segment_sum
        num_nodes = node_features.shape[0]
        aggregated = jax.ops.segment_sum(
            messages, receivers, num_segments=num_nodes
        )  # (N, hidden_dim)

        # Update node features
        update_input = jnp.concatenate([node_features, aggregated], axis=-1)
        updated = jax.vmap(self.update_mlp)(update_input)  # (N, node_dim)

        return updated


# ===============================================================================
# Edge readout head
# ===============================================================================

class EdgeReadout(eqx.Module):
    """Read out per-edge scalar predictions from node embeddings.

    Concatenates the sender embedding, receiver embedding, and edge features
    and maps them through an MLP with a ``softplus`` final activation to
    guarantee positive output.

    Parameters
    ----------
    mlp : eqx.nn.MLP
        The readout MLP producing a single scalar per edge.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        # Input: [h_sender, h_receiver, e_ij] -> 1
        self.mlp = eqx.nn.MLP(
            in_size=2 * node_dim + edge_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.elu,
            final_activation=jax.nn.softplus,
            key=key,
        )

    def __call__(
        self,
        node_features: Float[Array, "N node_dim"],
        edge_index: Int[Array, "2 E"],
        edge_features: Float[Array, "E edge_dim"],
    ) -> Float[Array, " E"]:
        """Predict a positive scalar for every edge.

        Parameters
        ----------
        node_features : Array
            Node embeddings ``(N, node_dim)``.
        edge_index : Array
            COO edge indices ``(2, E)``.
        edge_features : Array
            Edge features ``(E, edge_dim)``.

        Returns
        -------
        Array
            Per-edge scalars of shape ``(E,)``.
        """
        senders = edge_index[0]
        receivers = edge_index[1]

        h_senders = node_features[senders]
        h_receivers = node_features[receivers]

        edge_input = jnp.concatenate(
            [h_senders, h_receivers, edge_features], axis=-1
        )
        q_per_edge = jax.vmap(self.mlp)(edge_input)  # (E, 1)

        return q_per_edge.squeeze(-1)  # (E,)


# ===============================================================================
# GNN Encoder
# ===============================================================================

class GNNEncoder(Encoder):
    """Graph Neural Network encoder for variable-topology form-finding.

    Uses message passing to learn node embeddings from mesh topology, then
    reads out per-edge force density values.  The final output applies the
    same sign/shift convention as :class:`~neural_fdm.models.MLPEncoder`::

        q = (q_hat + q_shift) * edges_signs

    Parameters
    ----------
    edges_signs : Array
        ``+1`` for tension, ``-1`` for compression, per edge.
    q_shift : float
        Minimum force density magnitude.
    node_embed : eqx.nn.Linear
        Projects raw 3-D coordinates into the hidden space.
    layers : list of MessagePassingLayer
        Stack of message-passing layers.
    edge_readout : EdgeReadout
        Maps final node embeddings to per-edge positive scalars.
    _edge_index : Array
        Reference edge connectivity ``(2, E)`` stored at init time.
    """

    node_embed: eqx.nn.Linear
    layers: list
    edge_readout: EdgeReadout
    _edge_index: Array

    def __init__(
        self,
        edges_signs: Array,
        q_shift: float = 0.0,
        slice_out: bool = False,
        slice_indices: Array | None = None,
        node_feat_dim: int = 3,
        edge_feat_dim: int = 4,   # relative_pos (3) + distance (1)
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_index: Array | None = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(edges_signs, q_shift, slice_out, slice_indices)

        keys = jax.random.split(key, num_layers + 2)

        # Initial node embedding: (3,) -> (hidden_dim,)
        self.node_embed = eqx.nn.Linear(node_feat_dim, hidden_dim, key=keys[0])

        # Message-passing stack
        self.layers = []
        for i in range(num_layers):
            layer = MessagePassingLayer(
                node_dim=hidden_dim,
                edge_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                key=keys[i + 1],
            )
            self.layers.append(layer)

        # Edge readout head
        self.edge_readout = EdgeReadout(
            node_dim=hidden_dim,
            edge_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            key=keys[-1],
        )

        # Store the edge index (fixed for a given mesh topology)
        if edge_index is not None:
            self._edge_index = jnp.asarray(edge_index, dtype=jnp.int32)
        else:
            self._edge_index = jnp.zeros((2, 0), dtype=jnp.int32)

    def __call__(
        self,
        x: Float[Array, " N3"],
        edge_index: Int[Array, "2 E"] | None = None,
        structure=None,
    ) -> Float[Array, " E"]:
        """Predict force densities from vertex positions.

        Parameters
        ----------
        x : Array
            Flat vertex positions of length ``N * 3``.
        edge_index : Array, optional
            Edge connectivity ``(2, E)``.  If *None*, the stored
            ``_edge_index`` is used.
        structure : EquilibriumStructure, optional
            Not used directly but accepted for interface compatibility.

        Returns
        -------
        q : Array
            Signed force density per edge ``(E,)``.
        """
        # Reshape flat input to (N, 3)
        node_xyz = jnp.reshape(x, (-1, 3))

        # Select edge index
        ei = edge_index if edge_index is not None else self._edge_index

        # Compute edge features from current geometry
        relative_pos, distances = compute_edge_features(node_xyz, ei)
        edge_feat = jnp.concatenate([relative_pos, distances], axis=-1)  # (E, 4)

        # Embed raw node coordinates into hidden space
        node_h = jax.vmap(self.node_embed)(node_xyz)  # (N, hidden_dim)

        # Message-passing with residual connections
        for layer in self.layers:
            node_h = layer(node_h, ei, edge_feat) + node_h

        # Edge readout -> per-edge positive scalar
        q_hat = self.edge_readout(node_h, ei, edge_feat)  # (E,)

        # Apply sign and shift (same convention as MLPEncoder)
        return (q_hat + self.q_shift) * self.edges_signs
