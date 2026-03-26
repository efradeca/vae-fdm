"""Pure-numpy API for external tool integration.

This module provides a clean interface that accepts and returns only
numpy arrays and Python dicts, with no JAX objects in the public API.

Example
-------
>>> from neural_fdm.interop import predict_equilibrium
>>> result = predict_equilibrium(
...     vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5], [1, 1, 0.5]]),
...     edges=np.array([[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]]),
...     supports=np.array([0, 1]),
...     loads=np.array([[0, 0, 0], [0, 0, 0], [0, 0, -0.1], [0, 0, -0.1]]),
...     model_path="data/formfinder_bezier.eqx",
...     config_path="scripts/bezier.yml",
... )
>>> result["vertices"]      # (N, 3) predicted equilibrium positions
>>> result["force_densities"]  # (E,) per-edge force densities
>>> result["forces"]          # (E,) axial forces
>>> result["residuals"]       # (N, 3) force residuals
"""

from __future__ import annotations

import numpy as np


def predict_equilibrium(
    vertices: np.ndarray,
    edges: np.ndarray,
    supports: np.ndarray,
    loads: np.ndarray,
    model_path: str,
    config_path: str,
) -> dict[str, np.ndarray]:
    """Predict equilibrium shape using a trained neural FDM model.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Target vertex positions.
    edges : ndarray (E, 2)
        Edge connectivity (vertex index pairs).
    supports : ndarray (S,)
        Indices of fixed (support) vertices.
    loads : ndarray (N, 3)
        External load vectors per vertex.
    model_path : str
        Path to trained model file (.eqx).
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - "vertices": ndarray (N, 3) - predicted equilibrium positions
        - "force_densities": ndarray (E,) - force density per edge
        - "forces": ndarray (E,) - axial force per edge
        - "lengths": ndarray (E,) - member lengths
        - "residuals": ndarray (N, 3) - force residuals at vertices
        - "inference_time_ms": float - prediction time in milliseconds
    """
    import time

    import jax.numpy as jnp
    import jax.random as jrn
    import yaml

    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_neural_model,
    )
    from neural_fdm.helpers import (
        edges_forces,
        edges_lengths,
        edges_vectors,
        vertices_residuals_from_xyz,
    )
    from neural_fdm.serialization import load_model

    # Load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build infrastructure
    key = jrn.PRNGKey(config.get("seed", 0))
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)

    # Build and load model
    model_skeleton = build_neural_model("formfinder", config, generator, key)
    model = load_model(model_path, model_skeleton)

    # Prepare input
    xyz_flat = jnp.array(vertices.flatten())

    # Predict
    t0 = time.perf_counter()
    x_hat, (q, xyz_fixed, loads_jax) = model(xyz_flat, structure, aux_data=True)
    x_hat.block_until_ready()
    t1 = time.perf_counter()

    # Post-process
    xyz_pred = jnp.reshape(x_hat, (-1, 3))
    connectivity = structure.connectivity
    vectors = edges_vectors(xyz_pred, connectivity)
    lengths_arr = edges_lengths(vectors)
    forces_arr = edges_forces(q, lengths_arr)
    residuals = vertices_residuals_from_xyz(q, loads_jax, xyz_pred, structure)

    return {
        "vertices": np.array(xyz_pred),
        "force_densities": np.array(q),
        "forces": np.array(forces_arr),
        "lengths": np.array(lengths_arr),
        "residuals": np.array(residuals),
        "inference_time_ms": (t1 - t0) * 1000,
    }
