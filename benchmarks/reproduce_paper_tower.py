"""Reproduce Paper Results: Tower Task (Pastrana et al., ICLR 2025).

Evaluates the tower formfinder using the paper's loss function
(shape error on compression rings + height + residual + regularization).

Expected results (paper Table 1, formfinder on towers):
  L_shape: 1.4 +/- 0.4
  L_physics: 0.0 +/- 0.0

Usage:
    python benchmarks/reproduce_paper_tower.py
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
    build_loss_function,
    build_neural_model,
)
from neural_fdm.serialization import load_model


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "tower.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(90)
    mk, gk = jax.random.split(key)
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)

    model_path = os.path.join(DATA, "formfinder_tower.eqx")
    if not os.path.exists(model_path):
        print("ERROR: Tower model not found. Download from paper or train:")
        print("  python scripts/train.py formfinder tower")
        sys.exit(1)

    sk = build_neural_model("formfinder", config, gen, mk)
    mdl = load_model(model_path, sk)
    loss_fn = build_loss_function(config, gen)

    print("=" * 65)
    print("  Tower Task Reproduction (100 random shapes, seed=90)")
    print("=" * 65)

    batch = 100
    xyz_batch = vmap(gen)(jrn.split(gk, batch))
    _ = mdl(xyz_batch[0], st)  # JIT warmup

    shape_errs, total_losses, times_ms = [], [], []
    for i in range(batch):
        t0 = time.perf_counter()
        _ = mdl(xyz_batch[i], st)
        jnp.array(0.0).block_until_ready()
        times_ms.append((time.perf_counter() - t0) * 1000)

        _, terms = loss_fn(mdl, st, xyz_batch[i][None, :], aux_data=True)
        shape_errs.append(float(terms["shape error"]))
        total_losses.append(float(terms["loss"]))

    print(f"\n  {'Metric':<25} {'Ours':>15} {'Paper':>15} {'Match':>8}")
    print("  " + "-" * 65)
    print(f"  {'Shape error':<25} {np.mean(shape_errs):>8.1f}+/-{np.std(shape_errs):<5.1f} {'1.4+/-0.4':>15} {'Yes' if abs(np.mean(shape_errs) - 1.4) < 2 * np.std(shape_errs) else 'Check':>8}")
    print(f"  {'Total loss':<25} {np.mean(total_losses):>8.1f}+/-{np.std(total_losses):<5.1f} {'':>15}")
    print(f"  {'Time [ms]':<25} {np.mean(times_ms[1:]):>8.1f}+/-{np.std(times_ms[1:]):<5.1f} {'1.3+/-0.1':>15} {'Platform':>8}")
    print(f"  {'Test shapes':<25} {batch:>15} {100:>15} {'Exact':>8}")
    print("=" * 65)

    csv_path = os.path.join(DATA, "paper_reproduction_tower_results.csv")
    with open(csv_path, "w") as f:
        f.write("shape_index,shape_error,total_loss,inference_ms\n")
        for i in range(batch):
            f.write(f"{i},{shape_errs[i]:.4f},{total_losses[i]:.4f},{times_ms[i]:.2f}\n")
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
