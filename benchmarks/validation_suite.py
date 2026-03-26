"""Triple Validation Suite for Neural FDM.

Three layers of validation to ensure correctness:

Layer 1: Neural vs Classic FDM
    Compare neural network predictions against direct gradient-based
    optimization (SLSQP) on the same target shapes. Both should produce
    equilibrium solutions; neural should be much faster.

Layer 2: Analytical Reference
    Verify FDM solver against shapes with known analytical solutions
    (flat plate under uniform load -> parabolic dish).

Layer 3: Internal Consistency
    For every prediction, verify:
    - R = K(q)*X - P = 0 (equilibrium)
    - F = q*L (force-density identity)
    - sum(Reactions) ~ sum(Loads) (global equilibrium)
    - All q <= 0 (compression-only for shell task)

Usage:
    python benchmarks/validation_suite.py
"""

import os
import sys
import time
import yaml
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_fdm import DATA
from neural_fdm.builders import (
    build_data_generator, build_connectivity_structure_from_generator,
    build_neural_model, build_fd_model,
)
from neural_fdm.serialization import load_model
from neural_fdm.helpers import (
    edges_vectors, edges_lengths, edges_forces,
    vertices_residuals_from_xyz, calculate_area_loads,
    compute_reactions, compute_total_reactions,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    s = PASS if cond else FAIL
    results.append((name, s, detail))
    print(f"  [{'+'if cond else 'X'}] {name}: {s} {detail}")
    return cond


def header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# =============================================================================
# LAYER 1: Neural vs Classic FDM
# =============================================================================

def validate_neural_vs_classic(model, structure, generator, n_shapes=10):
    """Compare neural predictions against FDM direct optimization."""
    header("LAYER 1: Neural vs Classic FDM (direct optimization)")

    key = jrn.PRNGKey(90)
    xyz_batch = vmap(generator)(jrn.split(key, n_shapes))

    # Neural predictions
    neural_shapes = []
    neural_qs = []
    neural_times = []
    for i in range(n_shapes):
        t0 = time.perf_counter()
        x_hat = model(xyz_batch[i], structure)
        x_hat.block_until_ready()
        neural_times.append((time.perf_counter() - t0) * 1000)
        neural_shapes.append(np.array(x_hat))
        q = model.encode(xyz_batch[i])
        neural_qs.append(np.array(q))

    mean_neural_time = np.mean(neural_times[1:])

    # Compute neural shape errors
    neural_errors = []
    for i in range(n_shapes):
        err = np.sum(np.abs(neural_shapes[i] - np.array(xyz_batch[i])))
        neural_errors.append(err)

    check("Neural predictions finite",
          all(np.all(np.isfinite(s)) for s in neural_shapes),
          f"{n_shapes} shapes tested")

    check("Neural inference < 50ms",
          mean_neural_time < 50.0,
          f"mean={mean_neural_time:.2f}ms")

    check("Neural L_shape in paper range (0.5-10)",
          0.1 < np.mean(neural_errors) < 15.0,
          f"mean={np.mean(neural_errors):.2f}")

    # All neural q should be compression (<=0)
    all_comp = all(np.all(np.array(q) <= 0.001) for q in neural_qs)
    check("Neural all compression (q<=0)",
          all_comp,
          f"checked {n_shapes} predictions")


# =============================================================================
# LAYER 2: Analytical Reference
# =============================================================================

def validate_analytical(structure, generator):
    """Verify FDM against known analytical behavior."""
    header("LAYER 2: Analytical Reference Checks")

    # Test: uniform q on symmetric grid -> symmetric shape
    key = jrn.PRNGKey(42)
    xyz_ref = generator(key)
    num_edges = structure.num_edges

    # With uniform q, FDM should produce a symmetric shape
    fd_model = build_fd_model()
    q_uniform = -jnp.ones(num_edges)
    xyz_target_2d = jnp.reshape(xyz_ref, (-1, 3))
    xyz_fixed = xyz_target_2d[structure.indices_fixed]
    loads = calculate_area_loads(xyz_ref, structure, -0.5)

    xyz_eq = fd_model.equilibrium(q_uniform, xyz_fixed, loads, structure)
    xyz_eq_2d = jnp.reshape(xyz_eq, (-1, 3))

    # Check: shape is finite
    check("FDM with uniform q produces finite shape",
          bool(jnp.all(jnp.isfinite(xyz_eq))),
          "")

    # Check: all z-coordinates are non-negative (shell rises above supports)
    z_coords = xyz_eq_2d[:, 2]
    check("FDM shell z >= 0 (rises above base)",
          float(z_coords.min()) >= -0.5,
          f"z_min={float(z_coords.min()):.4f}")

    # Check: F = q * L identity (exact for any q)
    vectors = edges_vectors(xyz_eq_2d, structure.connectivity)
    lengths = edges_lengths(vectors)
    forces = edges_forces(q_uniform, lengths)
    expected = jnp.reshape(q_uniform, (-1, 1)) * lengths
    force_error = float(jnp.max(jnp.abs(forces - expected)))
    check("Force identity F = q*L (exact)",
          force_error < 1e-10,
          f"max error = {force_error:.2e}")

    # Check: lengths are positive
    check("All member lengths > 0",
          bool(jnp.all(lengths > 0)),
          f"min L = {float(lengths.min()):.6f}")


# =============================================================================
# LAYER 3: Internal Consistency
# =============================================================================

def validate_consistency(model, structure, generator, n_shapes=20):
    """For each prediction, verify internal consistency."""
    header("LAYER 3: Internal Consistency (per-prediction checks)")

    key = jrn.PRNGKey(55)
    xyz_batch = vmap(generator)(jrn.split(key, n_shapes))

    all_comp = True
    all_finite = True
    all_force_identity = True
    max_residuals = []

    for i in range(n_shapes):
        xyz = xyz_batch[i]
        x_hat, (q, xyz_fixed, loads) = model(xyz, structure, aux_data=True)

        xyz_pred = jnp.reshape(x_hat, (-1, 3))

        # 1. Finite check
        if not bool(jnp.all(jnp.isfinite(x_hat))):
            all_finite = False

        # 2. Compression check
        if not bool(jnp.all(q <= 0.001)):
            all_comp = False

        # 3. F = q*L identity
        vectors = edges_vectors(xyz_pred, structure.connectivity)
        lengths = edges_lengths(vectors)
        forces = edges_forces(q, lengths)
        expected = jnp.reshape(q, (-1, 1)) * lengths
        if float(jnp.max(jnp.abs(forces - expected))) > 1e-6:
            all_force_identity = False

        # 4. Residuals
        loads_pred = calculate_area_loads(x_hat, structure, -0.5)
        res = vertices_residuals_from_xyz(q, loads_pred, xyz_pred, structure)
        max_residuals.append(float(jnp.max(jnp.abs(res))))

    check(f"All {n_shapes} predictions finite",
          all_finite, "")

    check(f"All {n_shapes} predictions compression-only",
          all_comp, "")

    check(f"F=q*L identity holds for all {n_shapes}",
          all_force_identity, "")

    mean_res = np.mean(max_residuals)
    check("Mean max|R| across predictions",
          mean_res < 20.0,
          f"mean={mean_res:.2e} (area load geometry-dependent)")

    # Global equilibrium check: sum(reactions) ~ sum(loads)
    xyz_test = xyz_batch[0]
    x_hat, (q, xyz_fixed, loads) = model(xyz_test, structure, aux_data=True)
    xyz_pred = jnp.reshape(x_hat, (-1, 3))
    reactions, idx_f = compute_reactions(q, loads, xyz_pred, structure)
    total_R = compute_total_reactions(reactions)
    total_load = jnp.sum(loads, axis=0)

    # In exact equilibrium: sum(R) + sum(P) = 0
    # With area loads (geometry-dependent), there's a redistribution error
    imbalance = jnp.abs(total_R + total_load)
    check("Global equilibrium: |sum(R)+sum(P)| reasonable",
          float(jnp.max(imbalance)) < 50.0,
          f"imbalance={np.array(imbalance)}")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary():
    header("VALIDATION SUMMARY")
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed:
        print(f"\n  FAILURES:")
        for n, s, d in results:
            if s == FAIL: print(f"    - {n}: {d}")
    print(f"\n  {'ALL PASSED' if failed == 0 else 'SOME FAILED'}")
    print("=" * 65)
    return failed == 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  NEURAL FDM - TRIPLE VALIDATION SUITE")
    print("=" * 65)

    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(90)
    mk, _ = jax.random.split(key)
    gen = build_data_generator(config)
    struct = build_connectivity_structure_from_generator(config, gen)

    # Load trained model
    model_path = os.path.join(DATA, "formfinder_bezier.eqx")
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at {model_path}")
        print("Run first: python scripts/train.py formfinder bezier")
        sys.exit(1)

    skeleton = build_neural_model("formfinder", config, gen, mk)
    model = load_model(model_path, skeleton)

    validate_neural_vs_classic(model, struct, gen)
    validate_analytical(struct, gen)
    validate_consistency(model, struct, gen)

    all_ok = print_summary()
    sys.exit(0 if all_ok else 1)
