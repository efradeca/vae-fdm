"""Benchmark: VAE diversity and solution multiplicity.

Quantifies the VAE's ability to generate diverse equilibrium solutions
for the same target shape, and measures stability across random seeds.

Writes per-(target, seed) results to data/benchmark_vae_diversity.csv so
the metrics are reproducible and reviewable.

Usage:
    python benchmarks/benchmark_vae_diversity.py
"""

import csv
import os
import sys

import jax
import jax.random as jrn
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_fdm import DATA
from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_neural_model,
)
from neural_fdm.serialization import load_model
from neural_fdm.variational import compute_diversity_metrics


def main(n_targets=10, n_samples=20):
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "variational_bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(90)
    mk, gk = jax.random.split(key)
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)

    model_path = os.path.join(DATA, "variational_formfinder_variational_bezier.eqx")
    if not os.path.exists(model_path):
        print("ERROR: Train VAE model first")
        sys.exit(1)

    sk = build_neural_model("variational_formfinder", config, gen, mk)
    mdl = load_model(model_path, sk)

    print("=" * 65)
    print(f"  VAE Diversity Benchmark ({n_targets} targets, {n_samples} samples each)")
    print("=" * 65)

    all_diversity = []
    all_q_std = []
    rows = []

    seeds = [90, 42, 17]
    for seed in seeds:
        seed_key = jrn.PRNGKey(seed)
        dk, sk_gen = jax.random.split(seed_key)
        xyz_batch = jax.vmap(gen)(jrn.split(sk_gen, n_targets))

        diversities = []
        for i in range(n_targets):
            xyz = xyz_batch[i]
            sample_key = jrn.PRNGKey(seed * 1000 + i)
            x_hats, qs = mdl.sample(xyz, st, sample_key, num_samples=n_samples)
            metrics = compute_diversity_metrics(x_hats, qs)
            diversities.append(metrics["shape_pairwise_L1_mean"])
            q_std_mean = float(np.mean(np.array(metrics["q_std_per_edge"])))
            all_q_std.append(q_std_mean)
            rows.append({
                "task": "bezier",
                "seed": seed,
                "target_idx": i,
                "n_samples": n_samples,
                "shape_pairwise_L1_mean": float(metrics["shape_pairwise_L1_mean"]),
                "shape_pairwise_L1_std": float(metrics.get("shape_pairwise_L1_std", 0.0)),
                "q_std_mean": q_std_mean,
                "q_std_max": float(np.max(np.array(metrics["q_std_per_edge"]))),
            })

        mean_div = np.mean(diversities)
        all_diversity.append(mean_div)
        print(f"  Seed {seed}: mean diversity = {mean_div:.2f}")

    print(f"\n  Cross-seed diversity: {np.mean(all_diversity):.2f} +/- {np.std(all_diversity):.2f}")
    print(f"  Mean q_std per edge: {np.mean(all_q_std):.4f}")
    print(f"  Diversity > 0: {np.mean(all_diversity) > 0}")
    print("=" * 65)

    csv_path = os.path.join(DATA, "benchmark_vae_diversity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
