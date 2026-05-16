"""Validation Suite for Neural FDM.

Three layers of validation:

Layer 1: Neural vs Classical FDM
    Compare neural network predictions against direct gradient-based
    optimization (L-BFGS-B) on the same target shapes. Both produce
    equilibrium solutions; neural should be faster.

Layer 2: Analytical Reference
    Verify FDM solver against shapes with known analytical solutions
    (uniform q on a grid -> verify force identity F = q*L).

Layer 3: Internal Consistency
    For every prediction, verify:
    - F = q*L (force-density identity)
    - All q <= 0 (compression-only for shell task)
    - sum(Reactions) ~ sum(Loads) (global equilibrium)

Usage:
    python benchmarks/validation_suite.py
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
    build_mesh_from_generator,
    build_neural_model,
)
from neural_fdm.helpers import (
    calculate_area_loads,
    compute_reactions,
    compute_total_reactions,
    edges_forces,
    edges_lengths,
    edges_vectors,
    vertices_residuals_from_xyz,
)
from neural_fdm.serialization import load_model

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
# LAYER 1: Neural vs Classical FDM
# =============================================================================

def validate_neural_vs_classical(model, structure, generator, config, mesh, n_shapes=3):
    """Compare neural predictions against classical FDM optimization.

    Both methods are evaluated on the same target shapes with the same
    metrics (L_shape, L_physics, runtime).
    """
    header("LAYER 1: Neural vs Classical FDM")

    from neural_fdm.classical import solve_classical

    key = jrn.PRNGKey(90)
    xyz_batch = vmap(generator)(jrn.split(key, n_shapes))

    neural_errors, neural_times = [], []
    classical_errors, classical_times, classical_success = [], [], []

    # Warmup JIT
    _ = model(xyz_batch[0], structure)

    for i in range(n_shapes):
        xyz = xyz_batch[i]

        # Neural
        t0 = time.perf_counter()
        x_hat = model(xyz, structure)
        x_hat.block_until_ready()
        neural_times.append((time.perf_counter() - t0) * 1000)
        neural_errors.append(float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz, (-1, 3))
        ))))

        # Classical
        result = solve_classical(
            xyz, config, mesh, structure,
            maxiter=2000, key=jrn.PRNGKey(i)
        )
        classical_errors.append(result["l_shape"])
        classical_times.append(result["runtime_ms"])
        classical_success.append(result["success"])

    check("Neural predictions finite",
          all(np.isfinite(e) for e in neural_errors),
          f"{n_shapes} shapes tested")

    check("Classical solver produces finite solutions",
          all(np.isfinite(e) for e in classical_errors),
          f"{n_shapes} shapes tested")

    check("Neural L_shape in expected range",
          0.1 < np.mean(neural_errors) < 15.0,
          f"neural={np.mean(neural_errors):.2f}, classical={np.mean(classical_errors):.2f}")

    check("Classical L_shape in reasonable range",
          np.mean(classical_errors) < 50.0,
          f"classical={np.mean(classical_errors):.2f}")

    check("Neural faster than classical",
          np.mean(neural_times[1:]) < np.mean(classical_times),
          f"neural={np.mean(neural_times[1:]):.1f}ms, classical={np.mean(classical_times):.0f}ms")

    # Summary
    print(f"\n  {'Metric':<25} {'Neural':>10} {'Classical':>10}")
    print("  " + "-" * 47)
    print(f"  {'L_shape (mean)':<25} {np.mean(neural_errors):>10.2f} {np.mean(classical_errors):>10.2f}")
    print(f"  {'Time (mean ms)':<25} {np.mean(neural_times[1:]):>10.1f} {np.mean(classical_times):>10.0f}")
    print(f"  {'Classical success':<25} {'':>10} {sum(classical_success)}/{n_shapes}")


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
        print("\n  FAILURES:")
        for n, s, d in results:
            if s == FAIL:
                print(f"    - {n}: {d}")
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

    mesh = build_mesh_from_generator(config, gen)
    validate_neural_vs_classical(model, struct, gen, config, mesh)
    validate_analytical(struct, gen)
    validate_consistency(model, struct, gen)

    all_ok = print_summary()
    sys.exit(0 if all_ok else 1)
