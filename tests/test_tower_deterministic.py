"""Tests for tower task deterministic form-finding.

Verifies that the tower encoder-decoder pipeline produces valid
equilibrium solutions with mixed tension/compression forces.
"""

import jax.numpy as jnp
import jax.random as jrn
import pytest

from neural_fdm.builders import build_loss_function, build_neural_model


class TestTowerDeterministic:
    @pytest.fixture
    def tower_model(self, tower_config, tower_generator, prng_key):
        return build_neural_model("formfinder", tower_config, tower_generator, prng_key)

    def test_forward_finite(self, tower_model, tower_structure, tower_generator, prng_key):
        """Forward pass produces finite output."""
        xyz = tower_generator(prng_key)
        x_hat = tower_model(xyz, tower_structure)
        assert jnp.all(jnp.isfinite(x_hat))

    def test_mixed_tension_compression(self, tower_model, tower_structure, tower_generator, prng_key):
        """Tower produces both positive (tension) and negative (compression) q."""
        xyz = tower_generator(prng_key)
        q = tower_model.encode(xyz)
        assert float(jnp.min(q)) < -0.1, "Should have compression"
        assert float(jnp.max(q)) > 0.1, "Should have tension"

    def test_encoder_sensitivity(self, tower_model, tower_structure, tower_generator):
        """Different inputs produce different force densities."""
        xyz1 = tower_generator(jrn.PRNGKey(0))
        xyz2 = tower_generator(jrn.PRNGKey(99))
        q1 = tower_model.encode(xyz1)
        q2 = tower_model.encode(xyz2)
        assert float(jnp.sum(jnp.abs(q1 - q2))) > 0.01

    def test_loss_finite(self, tower_model, tower_structure, tower_config, tower_generator, prng_key):
        """Tower loss function produces finite scalar."""
        loss_fn = build_loss_function(tower_config, tower_generator)
        xyz = tower_generator(prng_key)
        _, terms = loss_fn(tower_model, tower_structure, xyz[None, :], aux_data=True)
        assert jnp.isfinite(terms["loss"])
        assert "shape error" in terms
        assert "residual error" in terms
