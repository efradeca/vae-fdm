"""Example 01: Train a formfinder model on the masonry shell (Bezier) task.

This example trains a neural network coupled with a differentiable FDM solver
to generate compression-only masonry shell geometries in real time.

The model learns to map target Bezier surface shapes to force density values
that produce mechanically valid (equilibrium) geometries.

Usage:
    python examples/01_train_shell.py
"""

import os
import sys
import yaml
import time

import jax
import jax.random as jrn

from neural_fdm import DATA
from neural_fdm.builders import (
    build_data_generator,
    build_connectivity_structure_from_generator,
    build_neural_model,
    build_optimizer,
    build_loss_function,
)
from neural_fdm.training import train_model
from neural_fdm.serialization import save_model
from neural_fdm.plotting import plot_losses


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Reduce training for this example (fast demo)
    config["training"]["steps"] = 500
    config["training"]["batch_size"] = 16

    print("=" * 60)
    print("  Neural FDM - Training Shell Formfinder")
    print("=" * 60)

    # Setup
    seed = config["seed"]
    key = jrn.PRNGKey(seed)
    model_key, train_key = jax.random.split(key)

    # Build components
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    model = build_neural_model("formfinder", config, generator, model_key)
    optimizer = build_optimizer(config)
    loss_fn = build_loss_function(config, generator)

    print(f"Structure: {structure.num_vertices} vertices, {structure.num_edges} edges")
    print(f"Model parameters: {sum(p.size for p in jax.tree.leaves(model) if hasattr(p, 'size'))}")
    print(f"Training for {config['training']['steps']} steps...")

    # Train
    t0 = time.time()
    model, loss_history = train_model(
        model=model,
        structure=structure,
        optimizer=optimizer,
        generator=generator,
        loss_fn=loss_fn,
        num_steps=config["training"]["steps"],
        batch_size=config["training"]["batch_size"],
        key=train_key,
    )
    t1 = time.time()

    print(f"\nTraining completed in {t1 - t0:.1f}s")
    print(f"Final loss: {loss_history[-1]['loss']:.4f}")
    print(f"Final shape error: {loss_history[-1]['shape error']:.4f}")
    print(f"Final residual error: {loss_history[-1]['residual error']:.6f}")

    # Save
    os.makedirs(DATA, exist_ok=True)
    save_path = os.path.join(DATA, "example_formfinder_bezier.eqx")
    save_model(save_path, model)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
