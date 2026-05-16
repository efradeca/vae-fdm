"""Benchmark: Neural vs Classical FDM Form-Finding.

Compares neural encoder (real-time inference) against direct
gradient-based optimization (L-BFGS-B) on the same target shapes.
Exports per-shape CSV for reproducible analysis.

Usage:
    python benchmarks/benchmark_neural_vs_classical.py [--n_shapes 20] [--maxiter 2000]
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
    build_mesh_from_generator,
    build_neural_model,
)
from neural_fdm.classical import solve_classical
from neural_fdm.helpers import compute_l_physics
from neural_fdm.serialization import load_model


def main(n_shapes=20, maxiter=2000, seed=90):
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(seed)
    mk, gk = jax.random.split(key)
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)
    mesh = build_mesh_from_generator(config, gen)
    free_indices = sorted(list(mesh.vertices_free()))

    # Load neural model
    model_path = os.path.join(DATA, "formfinder_bezier.eqx")
    if not os.path.exists(model_path):
        print("ERROR: Train model first: python scripts/train.py formfinder bezier")
        sys.exit(1)
    sk = build_neural_model("formfinder", config, gen, mk)
    mdl = load_model(model_path, sk)

    # Generate test shapes
    xyz_batch = vmap(gen)(jrn.split(gk, n_shapes))
    _ = mdl(xyz_batch[0], st)  # JIT warmup

    print("=" * 65)
    print(f"  Neural vs Classical Benchmark ({n_shapes} shapes, seed={seed})")
    print("=" * 65)

    rows = []
    for i in range(n_shapes):
        xyz = xyz_batch[i]

        # Neural
        t0 = time.perf_counter()
        x_hat, (q, _, loads) = mdl(xyz, st, aux_data=True)
        x_hat.block_until_ready()
        t_neural = (time.perf_counter() - t0) * 1000

        l_shape_n = float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz, (-1, 3))
        )))
        l_physics_n = compute_l_physics(x_hat, q, loads, st, free_indices)

        rows.append({
            "shape_id": i, "method": "neural",
            "l_shape": l_shape_n, "l_physics": l_physics_n,
            "runtime_ms": t_neural, "success": True, "n_iters": 1,
        })

        # Classical
        result = solve_classical(
            xyz, config, mesh, st,
            maxiter=maxiter, key=jrn.PRNGKey(i + 100)
        )
        rows.append({
            "shape_id": i, "method": "classical",
            "l_shape": result["l_shape"], "l_physics": result["l_physics"],
            "runtime_ms": result["runtime_ms"],
            "success": result["success"], "n_iters": result["n_iters"],
        })

        if i % 5 == 0:
            print(f"  Shape {i}/{n_shapes}: neural={l_shape_n:.2f}, classical={result['l_shape']:.2f}")

    # Export CSV
    csv_path = os.path.join(DATA, "benchmark_neural_vs_classical.csv")
    with open(csv_path, "w") as f:
        f.write("shape_id,method,l_shape,l_physics,runtime_ms,success,n_iters\n")
        for r in rows:
            f.write(f"{r['shape_id']},{r['method']},{r['l_shape']:.4f},"
                    f"{r['l_physics']:.2e},{r['runtime_ms']:.2f},"
                    f"{r['success']},{r['n_iters']}\n")
    print(f"\n  Results saved to {csv_path}")

    # Summary
    neural_rows = [r for r in rows if r["method"] == "neural"]
    classical_rows = [r for r in rows if r["method"] == "classical"]

    print(f"\n  {'Metric':<25} {'Neural':>10} {'Classical':>10}")
    print("  " + "-" * 47)
    print(f"  {'L_shape (mean)':<25} {np.mean([r['l_shape'] for r in neural_rows]):>10.2f} "
          f"{np.mean([r['l_shape'] for r in classical_rows]):>10.2f}")
    print(f"  {'L_physics (mean)':<25} {np.mean([r['l_physics'] for r in neural_rows]):>10.2e} "
          f"{np.mean([r['l_physics'] for r in classical_rows]):>10.2e}")
    print(f"  {'Time (mean ms)':<25} {np.mean([r['runtime_ms'] for r in neural_rows]):>10.1f} "
          f"{np.mean([r['runtime_ms'] for r in classical_rows]):>10.0f}")
    print(f"  {'Speedup':<25} {np.mean([r['runtime_ms'] for r in classical_rows]) / max(np.mean([r['runtime_ms'] for r in neural_rows]), 0.01):>10.0f}x")

    # Paired bootstrap CI on the difference
    n_boot = 10000
    diffs = np.array(
        [n["l_shape"] - c["l_shape"]
         for n, c in zip(neural_rows, classical_rows)]
    )
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot_means[b] = diffs[idx].mean()
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    significant = (ci_low > 0) or (ci_high < 0)
    print(f"\n  Paired bootstrap (10k resamples) on L_shape(neural-classical):")
    print(f"    mean diff = {diffs.mean():+.3f}, 95% CI = [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"    significant at 95%: {significant}")
    print("=" * 65)

    # Augment CSV with summary row of CI stats
    summary_path = os.path.join(DATA, "benchmark_neural_vs_classical_ci.csv")
    with open(summary_path, "w") as f:
        f.write("metric,mean_diff,ci_low,ci_high,significant_at_95\n")
        f.write(f"l_shape_neural_minus_classical,{diffs.mean():.4f},"
                f"{ci_low:.4f},{ci_high:.4f},{significant}\n")
    print(f"  CI summary saved to {summary_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
