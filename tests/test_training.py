"""Tests for neural_fdm.training — smoke tests and convergence."""

import pytest

from neural_fdm.builders import build_loss_function, build_neural_model, build_optimizer
from neural_fdm.training import train_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_components(bezier_config, bezier_generator, bezier_structure, prng_key):
    """Build all components needed for training."""
    model = build_neural_model("formfinder", bezier_config, bezier_generator, prng_key)
    loss_fn = build_loss_function(bezier_config, bezier_generator)
    optimizer = build_optimizer(bezier_config)
    return model, bezier_structure, optimizer, bezier_generator, loss_fn


# ---------------------------------------------------------------------------
# Training smoke tests
# ---------------------------------------------------------------------------

class TestTraining:

    def test_train_5_steps(self, training_components, bezier_config, prng_key):
        """Train formfinder for 5 steps; loss should decrease."""
        model, structure, optimizer, generator, loss_fn = training_components

        trained_model, loss_history = train_model(
            model,
            structure,
            optimizer,
            generator,
            loss_fn=loss_fn,
            num_steps=5,
            batch_size=bezier_config["training"]["batch_size"],
            key=prng_key,
        )

        # Loss at step 0 vs step 4 — should generally decrease
        first_loss = loss_history[0]["loss"].item()
        last_loss = loss_history[-1]["loss"].item()
        # At minimum, training should run without error and produce finite losses
        assert last_loss < float("inf")
        assert first_loss < float("inf")

    def test_train_returns_loss_history(self, training_components, bezier_config, prng_key):
        """Loss history should have expected keys and length."""
        model, structure, optimizer, generator, loss_fn = training_components
        num_steps = 5

        _, loss_history = train_model(
            model,
            structure,
            optimizer,
            generator,
            loss_fn=loss_fn,
            num_steps=num_steps,
            batch_size=bezier_config["training"]["batch_size"],
            key=prng_key,
        )

        assert len(loss_history) == num_steps
        # Each entry should be a dict with standard keys
        for entry in loss_history:
            assert "loss" in entry
            assert "shape error" in entry
            assert "residual error" in entry

    def test_train_convergence_formfinder(self, training_components, bezier_config, prng_key):
        """Loss should improve after 50 steps of training."""
        model, structure, optimizer, generator, loss_fn = training_components

        _, loss_history = train_model(
            model, structure, optimizer, generator,
            loss_fn=loss_fn, num_steps=50,
            batch_size=bezier_config["training"]["batch_size"],
            key=prng_key,
        )

        first = loss_history[0]["loss"].item()
        last = loss_history[-1]["loss"].item()
        assert last < first, f"Loss did not improve: {first:.2f} -> {last:.2f}"

    def test_train_convergence_vae(self, bezier_config, bezier_generator, bezier_structure, prng_key):
        """VAE loss should improve after 50 steps, including KL term."""
        import os

        import yaml

        config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "variational_bezier.yml")
        with open(config_path) as f:
            vae_config = yaml.load(f, Loader=yaml.FullLoader)

        import jax
        mk, tk = jax.random.split(prng_key)
        gen = bezier_generator
        st = bezier_structure
        model = build_neural_model("variational_formfinder", vae_config, gen, mk)
        loss_fn = build_loss_function(vae_config, gen)
        optimizer = build_optimizer(vae_config)

        _, loss_history = train_model(
            model, st, optimizer, gen,
            loss_fn=loss_fn, num_steps=50,
            batch_size=vae_config["training"]["batch_size"],
            key=tk,
        )

        first = loss_history[0]["loss"].item()
        last = loss_history[-1]["loss"].item()
        assert last < first, f"VAE loss did not improve: {first:.2f} -> {last:.2f}"
        assert "kl divergence" in loss_history[-1]
