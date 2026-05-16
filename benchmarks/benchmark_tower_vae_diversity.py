"""Benchmark: Tower VAE diversity and solution multiplicity.

Quantifies the tower VAE's ability to generate diverse equilibrium
solutions. Note: Tower VAE has higher shape error than the deterministic
model (~85 vs ~0.8) due to the inherent VAE reconstruction-diversity
trade-off with 656 latent dimensions.

Usage:
    python benchmarks/benchmark_tower_vae_diversity.py
"""

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


def main(n_targets=5, n_samples=10):
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "variational_tower.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(90)
    mk, gk = jax.random.split(key)
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)

    model_path = os.path.join(DATA, "variational_formfinder_variational_tower.eqx")
    if not os.path.exists(model_path):
        print("ERROR: Tower VAE model not found. Train with:")
        print("  python scripts/train.py variational_formfinder variational_tower")
        sys.exit(1)

    sk = build_neural_model("variational_formfinder", config, gen, mk)
    mdl = load_model(model_path, sk)

    print("=" * 65)
    print(f"  Tower VAE Diversity ({n_targets} targets, {n_samples} samples)")
    print("  Note: Tower VAE has higher shape error than deterministic")
    print("  due to 656-dim latent space. Diversity is the key metric.")
    print("=" * 65)

    seeds = [90, 42, 17]
    all_diversity = []
    for seed in seeds:
        sk_gen = jrn.PRNGKey(seed)
        _, dk = jax.random.split(sk_gen)
        xyz_batch = jax.vmap(gen)(jrn.split(dk, n_targets))

        diversities = []
        for i in range(n_targets):
            x_hats, qs = mdl.sample(xyz_batch[i], st, jrn.PRNGKey(seed * 1000 + i), num_samples=n_samples)
            metrics = compute_diversity_metrics(x_hats, qs)
            diversities.append(metrics["shape_pairwise_L1_mean"])

        mean_div = np.mean(diversities)
        all_diversity.append(mean_div)
        print(f"  Seed {seed}: mean diversity = {mean_div:.2f}")

    print(f"\n  Cross-seed diversity: {np.mean(all_diversity):.2f} +/- {np.std(all_diversity):.2f}")
    print(f"  Diversity > 0: {np.mean(all_diversity) > 0}")
    print("=" * 65)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
