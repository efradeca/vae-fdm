"""Tests for neural_fdm.serialization — save/load round-trip."""

import jax.numpy as jnp
import jax.random as jrn

from neural_fdm.builders import build_neural_model
from neural_fdm.serialization import load_model, save_model

# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_save_load_roundtrip(
        self, bezier_config, bezier_generator, bezier_structure, prng_key, tmp_path
    ):
        """Save a model, load it back, and verify predictions match."""
        model = build_neural_model(
            "formfinder", bezier_config, bezier_generator, prng_key
        )

        # Generate a test sample
        sample_x = bezier_generator(prng_key)

        # Get predictions before saving
        x_hat_before = model(sample_x, bezier_structure)

        # Save and load
        filepath = str(tmp_path / "test_model.eqx")
        save_model(filepath, model)

        # Build a fresh skeleton with a different key to ensure weights differ
        key2 = jrn.PRNGKey(999)
        model_skeleton = build_neural_model(
            "formfinder", bezier_config, bezier_generator, key2
        )
        loaded_model = load_model(filepath, model_skeleton)

        # Get predictions after loading
        x_hat_after = loaded_model(sample_x, bezier_structure)

        assert jnp.allclose(x_hat_before, x_hat_after, atol=1e-6)
