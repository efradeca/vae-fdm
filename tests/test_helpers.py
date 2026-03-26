"""Tests for neural_fdm.helpers — FDM math functions."""

import jax.numpy as jnp

from neural_fdm.helpers import (
    calculate_area_loads,
    calculate_equilibrium_state,
    edges_forces,
    edges_lengths,
    edges_vectors,
)

# ---------------------------------------------------------------------------
# edges_vectors
# ---------------------------------------------------------------------------

class TestEdgesVectors:

    def test_edges_vectors_shape(self, small_structure, small_xyz):
        """Output shape must be (num_edges, 3)."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)

        num_edges = small_structure.num_edges
        assert vecs.shape == (num_edges, 3)

    def test_edges_vectors_nonzero(self, small_structure, small_xyz):
        """At least some edge vectors should be nonzero."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)
        assert jnp.any(vecs != 0.0)


# ---------------------------------------------------------------------------
# edges_lengths
# ---------------------------------------------------------------------------

class TestEdgesLengths:

    def test_edges_lengths_positive(self, small_structure, small_xyz):
        """All edge lengths must be strictly positive."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)
        lengths = edges_lengths(vecs)

        assert jnp.all(lengths > 0.0)

    def test_edges_lengths_shape(self, small_structure, small_xyz):
        """Lengths shape should be (num_edges, 1)."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)
        lengths = edges_lengths(vecs)

        num_edges = small_structure.num_edges
        assert lengths.shape == (num_edges, 1)


# ---------------------------------------------------------------------------
# edges_forces
# ---------------------------------------------------------------------------

class TestEdgesForces:

    def test_edges_forces_sign(self, small_structure, small_xyz, small_q):
        """Force signs should match q signs (negative q -> negative forces)."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)
        lengths = edges_lengths(vecs)
        forces = edges_forces(small_q, lengths)

        # q is all -1, lengths are positive, so forces should be all negative
        assert jnp.all(forces < 0.0)

    def test_edges_forces_shape(self, small_structure, small_xyz, small_q):
        """Forces shape should be (num_edges, 1)."""
        connectivity = small_structure.connectivity
        xyz = jnp.reshape(small_xyz, (-1, 3))
        vecs = edges_vectors(xyz, connectivity)
        lengths = edges_lengths(vecs)
        forces = edges_forces(small_q, lengths)

        num_edges = small_structure.num_edges
        assert forces.shape == (num_edges, 1)


# ---------------------------------------------------------------------------
# calculate_equilibrium_state
# ---------------------------------------------------------------------------

class TestCalculateEquilibriumState:

    def test_equilibrium_state_fields(self, small_structure, small_xyz, small_q, small_loads):
        """EquilibriumState must expose xyz, residuals, forces, lengths, loads, vectors."""
        xyz = jnp.reshape(small_xyz, (-1, 3))
        state = calculate_equilibrium_state(small_q, xyz, small_loads, small_structure)

        assert state.xyz is not None
        assert state.residuals is not None
        assert state.forces is not None
        assert state.lengths is not None
        assert state.loads is not None
        assert state.vectors is not None

    def test_equilibrium_state_xyz_shape(self, small_structure, small_xyz, small_q, small_loads):
        """State xyz should have shape (num_vertices, 3)."""
        xyz = jnp.reshape(small_xyz, (-1, 3))
        state = calculate_equilibrium_state(small_q, xyz, small_loads, small_structure)
        num_vertices = small_structure.num_vertices
        assert state.xyz.shape == (num_vertices, 3)


# ---------------------------------------------------------------------------
# calculate_area_loads
# ---------------------------------------------------------------------------

class TestCalculateAreaLoads:

    def test_area_loads_sum(self, small_structure, small_xyz):
        """Total vertical load should approximately equal area * load_magnitude.

        The mesh is a unit quad (area ~1), load = -1.0, so the total z-load
        summed over all vertices should be close to -1.0.
        """
        load_per_area = -1.0
        loads = calculate_area_loads(small_xyz, small_structure, load_per_area)

        total_z_load = jnp.sum(loads[:, 2])
        # The quad has area ~1.0, so total z-load should be near -1.0
        assert jnp.abs(total_z_load - load_per_area) < 0.5

    def test_area_loads_shape(self, small_structure, small_xyz):
        """Output shape should be (num_vertices, 3)."""
        loads = calculate_area_loads(small_xyz, small_structure, -1.0)
        num_vertices = small_structure.num_vertices
        assert loads.shape == (num_vertices, 3)

    def test_area_loads_xy_near_zero(self, small_structure, small_xyz):
        """X and Y components of area loads should be near zero for a flat-ish mesh."""
        loads = calculate_area_loads(small_xyz, small_structure, -1.0)
        assert jnp.allclose(loads[:, 0], 0.0, atol=1e-3)
        assert jnp.allclose(loads[:, 1], 0.0, atol=1e-3)
