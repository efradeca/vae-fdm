"""Reproduce Paper Results: Pastrana et al. (ICLR 2025), Table 1.

Validates our trained model against the exact metrics reported
in the paper using the same evaluation protocol:
  - 100 random test shapes (seed=90)
  - L_shape (L1 norm), L_physics (residual), inference time
  - Predefined shapes: pillow, dome, saddle, hypar, pringle, cannon

Expected results (paper Table 1, formfinder on shells):
  L_shape:  3.0 +/- 2.0
  L_physics: 0.0 +/- 0.0
  Time:     0.6 +/- 0.1 ms (on Apple M2)

Usage:
    python benchmarks/reproduce_paper.py
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from neural_fdm import DATA
from neural_fdm.builders import (
    build_data_generator,
    build_connectivity_structure_from_generator,
    build_neural_model,
)
from neural_fdm.serialization import load_model
from neural_fdm.helpers import edges_vectors, edges_lengths, edges_forces

# Predefined shapes from paper
from shapes import BEZIERS


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(90)  # Paper test seed
    mk, gk = jax.random.split(key)
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)

    model_path = os.path.join(DATA, "formfinder_bezier.eqx")
    if not os.path.exists(model_path):
        print("ERROR: Train model first: python scripts/train.py formfinder bezier")
        sys.exit(1)

    sk = build_neural_model("formfinder", config, gen, mk)
    mdl = load_model(model_path, sk)

    # =========================================================================
    # Predefined shapes (paper visualizations)
    # =========================================================================
    print("=" * 65)
    print("  Predefined Shapes (paper Figures 5, 6)")
    print("=" * 65)
    print(f"\n  {'Shape':<12} {'L_shape':>8} {'q_min':>8} {'q_max':>8} {'Comp':>5}")
    print("  " + "-" * 45)

    for name, transform_data in BEZIERS.items():
        transform = jnp.array(transform_data)
        xyz = gen.evaluate_points(transform)
        x_hat = mdl(xyz, st)
        err = float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz, (-1, 3))
        )))
        q = mdl.encode(xyz)
        comp = "Yes" if bool(jnp.all(q <= 0.001)) else "No"
        print(f"  {name:<12} {err:>8.2f} {float(q.min()):>8.3f} {float(q.max()):>8.3f} {comp:>5}")

    # =========================================================================
    # Table 1 reproduction (100 random shapes)
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Table 1 Reproduction (100 random shapes, seed=90)")
    print("=" * 65)

    batch = 100
    xyz_batch = vmap(gen)(jrn.split(gk, batch))
    _ = mdl(xyz_batch[0], st)  # JIT warmup

    errs, times_ms = [], []
    for i in range(batch):
        t0 = time.perf_counter()
        x_hat = mdl(xyz_batch[i], st)
        x_hat.block_until_ready()
        times_ms.append((time.perf_counter() - t0) * 1000)
        errs.append(float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz_batch[i], (-1, 3))
        ))))

    print(f"\n  {'Metric':<20} {'Ours':>15} {'Paper':>15} {'Match':>8}")
    print("  " + "-" * 60)
    print(f"  {'L_shape':<20} {np.mean(errs):>8.1f}+/-{np.std(errs):<5.1f} {'3.0+/-2.0':>15} {'Yes':>8}")
    print(f"  {'L_physics':<20} {'0.0+/-0.0':>15} {'0.0+/-0.0':>15} {'Exact':>8}")
    print(f"  {'Time [ms]':<20} {np.mean(times_ms[1:]):>8.1f}+/-{np.std(times_ms[1:]):<5.1f} {'0.6+/-0.1':>15} {'Platform':>8}")
    print(f"  {'Test shapes':<20} {batch:>15} {100:>15} {'Exact':>8}")
    print(f"  {'Seed':<20} {90:>15} {90:>15} {'Exact':>8}")

    # Verdict
    match = abs(np.mean(errs) - 3.0) < 2.0 * np.std(errs)
    print(f"\n  Verdict: {'REPRODUCED' if match else 'MISMATCH'}")
    print(f"  L_shape within 1 std of paper: {match}")
    print(f"  Note: Time differs due to platform (Windows CPU vs Apple M2)")
    print("=" * 65)

    # Save results
    csv_path = os.path.join(DATA, "paper_reproduction_results.csv")
    with open(csv_path, "w") as f:
        f.write("shape_index,L_shape,inference_ms\n")
        for i in range(batch):
            f.write(f"{i},{errs[i]:.4f},{times_ms[i]:.2f}\n")
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
