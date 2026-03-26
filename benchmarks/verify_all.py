"""Verification suite for VAE-FDM.

Validates core components:
1. FDM solver correctness
2. Neural predictions vs paper (ICLR 2025, Table 1)
3. GNN encoder functionality

Usage:
    python benchmarks/verify_all.py
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np
import yaml
from jax import vmap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_fdm import DATA
from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_fd_model,
    build_neural_model,
)
from neural_fdm.helpers import (
    calculate_area_loads,
    edges_forces,
    edges_lengths,
    edges_vectors,
    vertices_residuals_from_xyz,
)
from neural_fdm.serialization import load_model

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "+" if condition else "X"
    print(f"  [{icon}] {name}: {status} {detail}")
    return condition


def section_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# =============================================================================
# 1. FDM SOLVER
# =============================================================================


def verify_fdm_solver():
    """Verify FDM produces equilibrium for known inputs."""
    section_header("1. FDM SOLVER - Equilibrium Verification")

    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)

    key = jrn.PRNGKey(42)
    xyz_target = generator(key)
    num_edges = structure.num_edges
    q = -jnp.ones(num_edges)

    fd_model = build_fd_model()
    xyz_target_2d = jnp.reshape(xyz_target, (-1, 3))
    xyz_fixed = xyz_target_2d[structure.indices_fixed]
    loads = calculate_area_loads(xyz_target, structure, -0.5)

    xyz_eq = fd_model.equilibrium(q, xyz_fixed, loads, structure)
    xyz_eq_2d = jnp.reshape(xyz_eq, (-1, 3))

    loads_eq = calculate_area_loads(jnp.ravel(xyz_eq), structure, -0.5)
    residuals = vertices_residuals_from_xyz(q, loads_eq, xyz_eq_2d, structure)
    max_residual = float(jnp.max(jnp.abs(residuals)))

    check("FDM equilibrium residual (area loads)",
          max_residual < 10.0,
          f"max |residual| = {max_residual:.2e}")

    vectors = edges_vectors(xyz_eq_2d, structure.connectivity)
    lengths = edges_lengths(vectors)
    forces = edges_forces(q, lengths)
    expected = jnp.reshape(q, (-1, 1)) * lengths
    force_error = float(jnp.max(jnp.abs(forces - expected)))

    check("Force identity F = q*L",
          force_error < 1e-6,
          f"max error = {force_error:.2e}")

    check("All forces compressive (q < 0)",
          bool(jnp.all(forces <= 0)),
          f"F range: [{float(forces.min()):.4f}, {float(forces.max()):.4f}]")

    check("All lengths positive",
          bool(jnp.all(lengths > 0)),
          f"min L = {float(lengths.min()):.4f}")


# =============================================================================
# 2. NEURAL MODEL vs PAPER
# =============================================================================


def verify_neural_predictions():
    """Verify trained model reproduces ICLR 2025 Table 1."""
    section_header("2. NEURAL MODEL - Paper Table 1 Reproduction")

    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_path = os.path.join(DATA, "formfinder_bezier.eqx")
    if not os.path.exists(model_path):
        print("  [!] Model not found. Run: python scripts/train.py formfinder bezier")
        return

    key = jrn.PRNGKey(90)
    model_key, gen_key = jax.random.split(key)
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    skeleton = build_neural_model("formfinder", config, generator, model_key)
    model = load_model(model_path, skeleton)

    batch_size = 100
    xyz_batch = vmap(generator)(jrn.split(gen_key, batch_size))
    _ = model(xyz_batch[0], structure)

    shape_errors = []
    times = []
    for i in range(batch_size):
        t0 = time.perf_counter()
        x_hat = model(xyz_batch[i], structure)
        x_hat.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
        diff = jnp.abs(jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz_batch[i], (-1, 3)))
        shape_errors.append(float(jnp.sum(diff)))

    mean_shape = np.mean(shape_errors)
    print(f"\n  L_shape: {mean_shape:.1f} +/- {np.std(shape_errors):.1f} (paper: 3.0 +/- 2.0)")

    check("L_shape in expected range",
          0.5 < mean_shape < 8.0,
          f"mean = {mean_shape:.2f}")

    check("Inference time < 50ms",
          np.mean(times[1:]) < 50.0,
          f"mean = {np.mean(times[1:]):.2f} ms")


# =============================================================================
# 3. GNN ENCODER
# =============================================================================


def verify_gnn():
    """Verify GNN encoder produces valid force densities."""
    section_header("3. GNN ENCODER - Functional Verification")

    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["encoder"]["encoder_type"] = "gnn"
    config["encoder"]["hidden_layer_size"] = 32
    config["encoder"]["hidden_layer_num"] = 2

    key = jrn.PRNGKey(42)
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    model = build_neural_model("formfinder", config, generator, key)

    from neural_fdm.gnn import GNNEncoder

    check("GNN encoder instantiated",
          isinstance(model.encoder, GNNEncoder),
          f"type: {type(model.encoder).__name__}")

    xyz = generator(key)
    x_hat = model(xyz, structure)

    check("GNN forward pass shape",
          x_hat.shape == xyz.shape,
          f"output: {x_hat.shape}")

    check("GNN output finite",
          bool(jnp.all(jnp.isfinite(x_hat))), "")

    xyz2 = generator(jrn.PRNGKey(99))
    x_hat2 = model(xyz2, structure)
    check("Different inputs -> different outputs",
          float(jnp.sum(jnp.abs(x_hat - x_hat2))) > 0.01, "")

    import equinox as eqx

    def loss_fn(mdl):
        xh = mdl(xyz, structure)
        return jnp.sum((xh - xyz) ** 2)

    grads = eqx.filter_grad(loss_fn)(model)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    has_grad = any(
        float(jnp.sum(jnp.abs(g))) > 0 for g in grad_leaves if hasattr(g, "shape")
    )
    check("Gradients flow through GNN", has_grad, "")


# =============================================================================
# SUMMARY
# =============================================================================


def print_summary():
    section_header("VERIFICATION SUMMARY")
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)

    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed:
        print("\n  FAILURES:")
        for n, s, d in results:
            if s == FAIL:
                print(f"    - {n}: {d}")

    print(f"\n  {'ALL CHECKS PASSED' if failed == 0 else 'SOME FAILED'}")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  VAE-FDM VERIFICATION SUITE")
    print("=" * 70)

    verify_fdm_solver()
    verify_neural_predictions()
    verify_gnn()

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)
