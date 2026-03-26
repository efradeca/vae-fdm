"""Shared test fixtures for neural_fdm tests."""

import jax.numpy as jnp
import jax.random as jrn
import pytest
from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructure

from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_fd_model,
    build_mesh_from_generator,
)

# ---------------------------------------------------------------------------
# Minimal 4-node mesh for fast unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mesh():
    """A minimal 2x2 quad mesh with 4 vertices and 1 face.

    Topology:
        2 --- 3
        |     |
        0 --- 1

    Vertex 0 and 1 are supports (fixed).
    """
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 1.0, 0.5]]
    faces = [[0, 1, 3, 2]]
    mesh = FDMesh.from_vertices_and_faces(vertices, faces)
    mesh.vertices_supports([0, 1])
    return mesh


@pytest.fixture
def small_structure(small_mesh):
    """EquilibriumMeshStructure from the small 4-node mesh."""
    return EquilibriumMeshStructure.from_mesh(small_mesh)


@pytest.fixture
def small_q(small_structure):
    """Force density values for the small mesh edges (all compression)."""
    num_edges = small_structure.num_edges
    return -jnp.ones(num_edges)


@pytest.fixture
def small_xyz(small_structure):
    """Flat xyz array for the small mesh."""
    return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.5])


@pytest.fixture
def small_loads(small_structure):
    """Vertex loads for the small mesh (downward on free nodes)."""
    n = small_structure.num_vertices
    loads = jnp.zeros((n, 3))
    return loads.at[2, 2].set(-0.1).at[3, 2].set(-0.1)


# ---------------------------------------------------------------------------
# Bezier task fixtures (small, fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def bezier_config():
    """Minimal bezier task configuration for testing."""
    return {
        "seed": 42,
        "generator": {
            "name": "bezier_symmetric_double",
            "bounds": "saddle",
            "num_uv": 5,
            "size": 5.0,
            "num_points": 4,
            "lerp_factor": 0.5,
        },
        "fdm": {"load": -0.5},
        "encoder": {
            "shift": 0.0,
            "hidden_layer_size": 32,
            "hidden_layer_num": 2,
            "activation_fn_name": "elu",
            "final_activation_fn_name": "softplus",
        },
        "decoder": {
            "include_params_xl": True,
            "hidden_layer_size": 32,
            "hidden_layer_num": 2,
            "activation_fn_name": "elu",
        },
        "loss": {
            "shape": {"include": True, "weight": 1.0},
            "residual": {"include": True, "weight": 1.0},
        },
        "optimizer": {"name": "adam", "learning_rate": 1e-3, "clip_norm": 0.0},
        "training": {"steps": 5, "batch_size": 2},
    }


@pytest.fixture
def bezier_generator(bezier_config):
    """Data generator for the bezier test task."""
    return build_data_generator(bezier_config)


@pytest.fixture
def bezier_structure(bezier_config, bezier_generator):
    """EquilibriumMeshStructure for the bezier test task."""
    return build_connectivity_structure_from_generator(bezier_config, bezier_generator)


@pytest.fixture
def bezier_mesh(bezier_config, bezier_generator):
    """FDMesh for the bezier test task."""
    return build_mesh_from_generator(bezier_config, bezier_generator)


@pytest.fixture
def prng_key():
    """A fixed PRNG key for reproducible tests."""
    return jrn.PRNGKey(42)


@pytest.fixture
def fd_model():
    """Force density equilibrium model."""
    return build_fd_model()
