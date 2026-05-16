"""Pure-numpy API for external tool integration.

This module provides a clean interface that accepts and returns only
numpy arrays and Python dicts, with no JAX objects in the public API.

Note: The mesh topology, loads, and boundary conditions are defined
by the YAML config file, not by the caller. The vertices argument
provides the target shape that the model maps to force densities.

Example
-------
>>> from neural_fdm.interop import predict_equilibrium
>>> result = predict_equilibrium(
...     vertices=target_xyz,  # (N, 3) target positions
...     model_path="data/formfinder_bezier.eqx",
...     config_path="scripts/bezier.yml",
... )
>>> result["vertices"]      # (N, 3) predicted equilibrium positions
>>> result["force_densities"]  # (E,) per-edge force densities
"""

from __future__ import annotations

import numpy as np


def predict_equilibrium(
    vertices: np.ndarray,
    model_path: str,
    config_path: str,
    model_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Predict equilibrium shape using a trained neural FDM model.

    The mesh topology and loads are determined by the config file.
    The vertices provide the target shape for the encoder.
    Model type is auto-detected from config (VAE if loss.vae section
    exists) or can be overridden via model_name.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Target vertex positions.
    model_path : str
        Path to trained model file (.eqx).
    config_path : str
        Path to YAML configuration file.
    model_name : str, optional
        Model type ("formfinder" or "variational_formfinder").
        Auto-detected from config if not specified.

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
        edges_lengths,
        edges_vectors,
        vertices_residuals_from_xyz,
    )
    from neural_fdm.serialization import load_model

    # Load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build infrastructure from config
    key = jrn.PRNGKey(config.get("seed", 0))
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)

    # Auto-detect model type from config, or use explicit override
    if model_name is None:
        is_vae = "vae" in config.get("loss", {})
        model_name = "variational_formfinder" if is_vae else "formfinder"

    model_skeleton = build_neural_model(model_name, config, generator, key)
    model = load_model(model_path, model_skeleton)

    # Prepare input
    xyz_flat = jnp.array(vertices.flatten())

    # Predict
    t0 = time.perf_counter()
    x_hat, aux = model(xyz_flat, structure, aux_data=True)
    x_hat.block_until_ready()
    t1 = time.perf_counter()

    # Unpack aux_data: VAE returns ((q, xyz_fixed, loads), mu, log_sigma)
    # Deterministic returns (q, xyz_fixed, loads)
    from neural_fdm.variational import VariationalAutoEncoder

    if isinstance(model, VariationalAutoEncoder):
        (q, xyz_fixed, loads_jax), _mu, _log_sigma = aux
    else:
        q, xyz_fixed, loads_jax = aux

    # Post-process
    xyz_pred = jnp.reshape(x_hat, (-1, 3))
    vectors = edges_vectors(xyz_pred, structure.connectivity)
    lengths_arr = edges_lengths(vectors)
    forces_arr = q * jnp.ravel(lengths_arr)
    residuals = vertices_residuals_from_xyz(q, loads_jax, xyz_pred, structure)

    return {
        "vertices": np.array(xyz_pred),
        "force_densities": np.array(q),
        "forces": np.array(forces_arr),
        "lengths": np.array(jnp.ravel(lengths_arr)),
        "residuals": np.array(residuals),
        "inference_time_ms": (t1 - t0) * 1000,
    }
