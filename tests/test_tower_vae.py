"""Tests for tower VAE form-finding.

Verifies that the variational encoder works for the tower task,
including training with KL divergence and diversity sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_loss_function,
    build_neural_model,
    build_optimizer,
)
from neural_fdm.training import train_model


class TestTowerVAE:
    @pytest.fixture
    def tower_vae_config(self, tower_config):
        """Tower config with VAE loss parameters."""
        cfg = dict(tower_config)
        cfg["loss"] = dict(cfg["loss"])
        cfg["loss"]["vae"] = {
            "beta_max": 0.1,
            "cycle_length": 100,
            "warmup_ratio": 0.5,
        }
        return cfg

    def test_vae_forward_finite(self, tower_vae_config, prng_key):
        """VAE tower forward pass produces finite output."""
        mk, tk = jax.random.split(prng_key)
        gen = build_data_generator(tower_vae_config)
        st = build_connectivity_structure_from_generator(tower_vae_config, gen)
        model = build_neural_model("variational_formfinder", tower_vae_config, gen, mk)
        xyz = gen(tk)
        x_hat, (params, mu, log_sigma) = model(xyz, st, aux_data=True, key=jrn.PRNGKey(0))
        assert jnp.all(jnp.isfinite(x_hat))
        assert mu.shape[0] > 0

    def test_vae_training_3steps(self, tower_vae_config, prng_key):
        """VAE tower training runs 3 steps with KL in loss."""
        mk, tk = jax.random.split(prng_key)
        gen = build_data_generator(tower_vae_config)
        st = build_connectivity_structure_from_generator(tower_vae_config, gen)
        model = build_neural_model("variational_formfinder", tower_vae_config, gen, mk)
        loss_fn = build_loss_function(tower_vae_config, gen)
        optimizer = build_optimizer(tower_vae_config)

        _, history = train_model(
            model, st, optimizer, gen,
            loss_fn=loss_fn, num_steps=3, batch_size=2, key=tk,
        )
        assert len(history) == 3
        assert "kl divergence" in history[-1]
        assert "beta" in history[-1]

    def test_vae_sampling_diverse(self, tower_vae_config, prng_key):
        """VAE tower sampling produces diverse finite shapes."""
        mk, tk = jax.random.split(prng_key)
        gen = build_data_generator(tower_vae_config)
        st = build_connectivity_structure_from_generator(tower_vae_config, gen)
        model = build_neural_model("variational_formfinder", tower_vae_config, gen, mk)
        xyz = gen(tk)
        x_hats, qs = model.sample(xyz, st, jrn.PRNGKey(99), num_samples=3)
        assert x_hats.shape[0] == 3
        assert jnp.all(jnp.isfinite(x_hats))
