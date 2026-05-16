"""End-to-end test for VAE training via the CLI pipeline.

Verifies that the variational_formfinder model can be trained
through the same code path as the CLI (train.py), and that the
KL divergence term is computed during training.
"""

import os
import sys

import jax
import jax.random as jrn
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def vae_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "variational_bezier.yml")
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class TestVAECLI:
    def test_vae_training_computes_kl(self, vae_config):
        """VAE training loop computes KL divergence and beta annealing."""
        from neural_fdm.builders import (
            build_connectivity_structure_from_generator,
            build_data_generator,
            build_loss_function,
            build_neural_model,
            build_optimizer,
        )
        from neural_fdm.training import train_model

        key = jrn.PRNGKey(91)
        mk, tk = jax.random.split(key)
        gen = build_data_generator(vae_config)
        st = build_connectivity_structure_from_generator(vae_config, gen)
        model = build_neural_model("variational_formfinder", vae_config, gen, mk)
        loss_fn = build_loss_function(vae_config, gen)
        optimizer = build_optimizer(vae_config)

        _, history = train_model(
            model, st, optimizer, gen,
            loss_fn=loss_fn, num_steps=3, batch_size=4, key=tk
        )

        assert len(history) == 3
        for h in history:
            assert "kl divergence" in h, "KL divergence must be in loss terms"
            assert "beta" in h, "Beta must be in loss terms"
            assert float(h["loss"]) > 0

    def test_vae_loss_dispatches_correctly(self, vae_config):
        """build_loss_function returns compute_loss_shell_vae for VAE configs."""
        from neural_fdm.builders import build_data_generator, build_loss_function
        from neural_fdm.losses import compute_loss_shell_vae

        gen = build_data_generator(vae_config)
        loss_fn = build_loss_function(vae_config, gen)
        inner = loss_fn.keywords.get("loss_fn")
        assert inner is compute_loss_shell_vae, (
            f"Expected compute_loss_shell_vae, got {inner.__name__}"
        )
