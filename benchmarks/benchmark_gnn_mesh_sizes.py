"""Benchmark: GNN encoder across different mesh resolutions.

Demonstrates that the same GNN architecture can be instantiated for
different mesh sizes. Each model instance is built for a specific
topology (edge_index fixed at construction).

Usage:
    python benchmarks/benchmark_gnn_mesh_sizes.py [--steps 200]
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

from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_loss_function,
    build_mesh_from_generator,
    build_neural_model,
    build_optimizer,
)
from neural_fdm.training import train_model


def benchmark_resolution(num_uv, config, steps=200, seed=42):
    """Train and evaluate GNN on a specific mesh resolution."""
    config = dict(config)
    config["generator"] = dict(config["generator"])
    config["generator"]["num_uv"] = num_uv
    config["training"] = dict(config["training"])
    config["training"]["steps"] = steps
    config["training"]["batch_size"] = 8

    key = jrn.PRNGKey(seed)
    mk, tk, ek = jax.random.split(key, 3)

    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)
    mesh = build_mesh_from_generator(config, gen)
    model = build_neural_model("formfinder", config, gen, mk)
    loss_fn = build_loss_function(config, gen)
    optimizer = build_optimizer(config)

    t0 = time.time()
    model_out, history = train_model(
        model, st, optimizer, gen,
        loss_fn=loss_fn, num_steps=steps, batch_size=8, key=tk
    )
    train_time = time.time() - t0

    # Evaluate on 10 test shapes
    xyz_test = vmap(gen)(jrn.split(ek, 10))
    _ = model_out(xyz_test[0], st)  # warmup

    errors = []
    for i in range(10):
        x_hat = model_out(xyz_test[i], st)
        err = float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz_test[i], (-1, 3))
        )))
        errors.append(err)

    n_verts = mesh.number_of_vertices()
    n_edges = mesh.number_of_edges()

    return {
        "num_uv": num_uv,
        "vertices": n_verts,
        "edges": n_edges,
        "l_shape_mean": np.mean(errors),
        "l_shape_std": np.std(errors),
        "train_time_s": train_time,
        "final_loss": float(history[-1]["loss"]),
    }


def main(steps=200):
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "gnn_bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    resolutions = [6, 10, 16]

    print("=" * 65)
    print(f"  GNN Encoder: Mesh Resolution Benchmark ({steps} steps)")
    print("=" * 65)
    print("\n  Note: Each resolution uses a separate GNN model instance.")
    print("  The same architecture and weight structure works across sizes.\n")

    results = []
    for nu in resolutions:
        print(f"  Training {nu}x{nu} mesh...")
        r = benchmark_resolution(nu, config, steps=steps)
        results.append(r)
        print(f"    Vertices={r['vertices']}, Edges={r['edges']}, "
              f"L_shape={r['l_shape_mean']:.2f}, Time={r['train_time_s']:.1f}s")

    print(f"\n  {'Resolution':<12} {'Vertices':>8} {'Edges':>6} {'L_shape':>10} {'Train(s)':>9}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['num_uv']}x{r['num_uv']:<8} {r['vertices']:>8} {r['edges']:>6} "
              f"{r['l_shape_mean']:>8.2f}±{r['l_shape_std']:<4.1f} {r['train_time_s']:>8.1f}")
    print("=" * 65)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
