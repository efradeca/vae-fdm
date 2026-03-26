"""Benchmark: MLP vs GNN Encoder for Neural FDM.

Comparison of encoder architectures on the same FDM backend
with the same training data and evaluation protocol.

Compares:
- MLP encoder (paper original, Pastrana et al. ICLR 2025)
- GNN encoder (message passing, this work)

Metrics:
- L_shape (shape matching accuracy)
- L_physics (equilibrium residual, should be ~0 for both)
- Inference time (ms)
- Parameter count
- Training time (s)

Usage:
    python benchmarks/benchmark_architectures.py [--steps 500] [--test_size 50]
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
    build_data_generator,
    build_connectivity_structure_from_generator,
    build_neural_model,
    build_optimizer,
    build_loss_function,
)
from neural_fdm.training import train_model
from neural_fdm.helpers import vertices_residuals_from_xyz, calculate_area_loads


def count_params(model):
    leaves = jax.tree_util.tree_leaves(model)
    return sum(x.size for x in leaves if hasattr(x, "size"))


def evaluate(model, structure, generator, key, test_size=50):
    """Evaluate model on test set. Returns dict of metrics."""
    xyz_batch = vmap(generator)(jrn.split(key, test_size))

    shape_errors = []
    max_residuals = []
    times = []

    # Warmup JIT
    _ = model(xyz_batch[0], structure)

    for i in range(test_size):
        xyz = xyz_batch[i]
        t0 = time.perf_counter()
        x_hat = model(xyz, structure)
        x_hat.block_until_ready()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

        shape_err = float(jnp.sum(jnp.abs(
            jnp.reshape(x_hat, (-1, 3)) - jnp.reshape(xyz, (-1, 3))
        )))
        shape_errors.append(shape_err)

        # Residual
        xyz_pred = jnp.reshape(x_hat, (-1, 3))
        loads = calculate_area_loads(x_hat, structure, -0.5)
        q = model.encode(xyz) if hasattr(model.encode, '__call__') else model.encoder(xyz)
        if isinstance(q, tuple):
            q = q[0]  # VAE returns (q, mu, log_sigma)
        res = vertices_residuals_from_xyz(q, loads, xyz_pred, structure)
        max_residuals.append(float(jnp.max(jnp.abs(res))))

    return {
        "L_shape_mean": np.mean(shape_errors),
        "L_shape_std": np.std(shape_errors),
        "L_physics_mean": np.mean(max_residuals),
        "inference_ms_mean": np.mean(times[1:]),  # skip JIT warmup
        "inference_ms_std": np.std(times[1:]),
    }


def main(steps=500, test_size=50):
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["training"]["steps"] = steps
    config["training"]["batch_size"] = 16
    config["encoder"]["hidden_layer_size"] = 64
    config["encoder"]["hidden_layer_num"] = 2

    key = jrn.PRNGKey(42)
    gen = build_data_generator(config)
    struct = build_connectivity_structure_from_generator(config, gen)

    results = {}

    # =========================================================================
    # MLP Encoder (paper original)
    # =========================================================================
    print("=" * 60)
    print("  Training MLP Encoder (paper original)")
    print("=" * 60)

    mk, tk, ek = jrn.split(key, 3)
    model_mlp = build_neural_model("formfinder", config, gen, mk)
    optimizer = build_optimizer(config)
    loss_fn = build_loss_function(config, gen)

    t0 = time.time()
    model_mlp, hist_mlp = train_model(
        model_mlp, struct, optimizer, gen,
        loss_fn=loss_fn, num_steps=steps, batch_size=16, key=tk
    )
    train_time_mlp = time.time() - t0

    metrics_mlp = evaluate(model_mlp, struct, gen, ek, test_size)
    metrics_mlp["params"] = count_params(model_mlp)
    metrics_mlp["train_time_s"] = train_time_mlp
    results["MLP"] = metrics_mlp

    # =========================================================================
    # GNN Encoder (this work)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Training GNN Encoder (this work)")
    print("=" * 60)

    config_gnn = dict(config)
    config_gnn["encoder"] = dict(config["encoder"])
    config_gnn["encoder"]["encoder_type"] = "gnn"
    config_gnn["encoder"]["hidden_layer_num"] = 3

    mk2, tk2, ek2 = jrn.split(jrn.PRNGKey(43), 3)
    model_gnn = build_neural_model("formfinder", config_gnn, gen, mk2)
    optimizer2 = build_optimizer(config_gnn)
    loss_fn2 = build_loss_function(config_gnn, gen)

    t0 = time.time()
    model_gnn, hist_gnn = train_model(
        model_gnn, struct, optimizer2, gen,
        loss_fn=loss_fn2, num_steps=steps, batch_size=16, key=tk2
    )
    train_time_gnn = time.time() - t0

    metrics_gnn = evaluate(model_gnn, struct, gen, ek2, test_size)
    metrics_gnn["params"] = count_params(model_gnn)
    metrics_gnn["train_time_s"] = train_time_gnn
    results["GNN"] = metrics_gnn

    # =========================================================================
    # Results Table
    # =========================================================================
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS: MLP vs GNN Encoder")
    print("=" * 60)
    print(f"\nTraining: {steps} steps, batch_size=16, hidden=64")
    print(f"Test set: {test_size} shapes\n")

    header = f"{'Metric':<25} {'MLP (paper)':>15} {'GNN (ours)':>15}"
    print(header)
    print("-" * len(header))

    for metric in ["L_shape_mean", "L_shape_std", "L_physics_mean",
                    "inference_ms_mean", "params", "train_time_s"]:
        v_mlp = results["MLP"][metric]
        v_gnn = results["GNN"][metric]
        if metric == "params":
            print(f"{metric:<25} {v_mlp:>15,} {v_gnn:>15,}")
        elif metric == "train_time_s":
            print(f"{metric:<25} {v_mlp:>15.1f} {v_gnn:>15.1f}")
        else:
            print(f"{metric:<25} {v_mlp:>15.4f} {v_gnn:>15.4f}")

    # Save to CSV
    csv_path = os.path.join(DATA, "benchmark_mlp_vs_gnn.csv")
    with open(csv_path, "w") as f:
        f.write("metric,MLP,GNN\n")
        for metric in results["MLP"]:
            f.write(f"{metric},{results['MLP'][metric]},{results['GNN'][metric]}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
