"""Example 03: Use a GNN encoder for topology-agnostic form-finding.

This example demonstrates training with a Graph Neural Network encoder
that can generalize across different mesh topologies, replacing the
fixed-topology MLP encoder.

Usage:
    python examples/03_gnn_encoder.py
"""

import os
import yaml
import time

import jax
import jax.numpy as jnp
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


def main():
    # Load base config
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Configure for GNN encoder
    config["encoder"]["encoder_type"] = "gnn"
    config["encoder"]["hidden_layer_size"] = 128
    config["encoder"]["hidden_layer_num"] = 4
    config["training"]["steps"] = 200
    config["training"]["batch_size"] = 8

    print("=" * 60)
    print("  Neural FDM - GNN Encoder Training Demo")
    print("=" * 60)

    # Setup
    key = jrn.PRNGKey(config["seed"])
    model_key, train_key = jax.random.split(key)

    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)

    # Build model with GNN encoder
    model = build_neural_model("formfinder", config, generator, model_key)
    optimizer = build_optimizer(config)
    loss_fn = build_loss_function(config, generator)

    encoder_type = type(model.encoder).__name__
    print(f"Encoder type: {encoder_type}")
    print(f"Structure: {structure.num_vertices} vertices, {structure.num_edges} edges")

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

    print(f"\nTraining: {config['training']['steps']} steps in {t1 - t0:.1f}s")
    print(f"Final loss: {loss_history[-1]['loss']:.4f}")

    # Test prediction
    test_key = jrn.PRNGKey(42)
    xyz_target = generator(test_key)
    x_hat = model(xyz_target, structure)

    shape_error = float(jnp.sum(jnp.abs(jnp.reshape(x_hat, (-1, 3)) -
                                         jnp.reshape(xyz_target, (-1, 3)))))
    print(f"Test shape error: {shape_error:.4f}")
    print(f"\nGNN encoder successfully predicts equilibrium shapes!")


if __name__ == "__main__":
    main()
