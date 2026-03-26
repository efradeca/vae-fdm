"""Tests for neural_fdm.models — model forward passes and gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from neural_fdm.builders import build_neural_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def formfinder_model(bezier_config, bezier_generator, prng_key):
    """Build a small formfinder model for testing."""
    return build_neural_model("formfinder", bezier_config, bezier_generator, prng_key)


@pytest.fixture
def sample_x(bezier_generator, prng_key):
    """Generate a single sample from the bezier generator."""
    return bezier_generator(prng_key)


@pytest.fixture
def batch_x(bezier_generator, prng_key):
    """Generate a small batch of samples."""
    keys = jrn.split(prng_key, 4)
    return jax.vmap(bezier_generator)(keys)


# ---------------------------------------------------------------------------
# MLPEncoder
# ---------------------------------------------------------------------------

class TestMLPEncoder:

    def test_mlp_encoder_output_shape(self, formfinder_model, sample_x, bezier_structure):
        """Encoder output should have shape (num_edges,)."""
        encoder = formfinder_model.encoder
        q = encoder(sample_x)
        num_edges = bezier_structure.num_edges
        assert q.shape == (num_edges,)

    def test_mlp_encoder_sign_constraint(self, formfinder_model, sample_x):
        """Output signs should match edges_signs (negative for compression edges)."""
        encoder = formfinder_model.encoder
        q = encoder(sample_x)
        edges_signs = encoder.edges_signs

        # Where edges_signs is -1, q should be <= 0; where +1, q should be >= 0
        signs_match = jnp.sign(q) == edges_signs
        # Allow zero values (sign(0) == 0 != edges_signs)
        is_zero = q == 0.0
        assert jnp.all(signs_match | is_zero)

    def test_mlp_encoder_deterministic(self, formfinder_model, sample_x):
        """Same input should produce the same output."""
        encoder = formfinder_model.encoder
        q1 = encoder(sample_x)
        q2 = encoder(sample_x)
        assert jnp.allclose(q1, q2)


# ---------------------------------------------------------------------------
# FDDecoder
# ---------------------------------------------------------------------------

class TestFDDecoder:

    def test_fd_decoder_equilibrium(self, formfinder_model, sample_x, bezier_structure):
        """FDDecoder output should satisfy equilibrium (residuals near zero)."""
        x_hat, (q, xyz_fixed, loads) = formfinder_model(
            sample_x, bezier_structure, aux_data=True
        )

        from neural_fdm.helpers import vertices_residuals_from_xyz

        residuals = vertices_residuals_from_xyz(q, loads, x_hat, bezier_structure)
        # Check residuals at free nodes only
        indices_free = bezier_structure.indices_free
        residuals_free = residuals[indices_free, :]
        assert jnp.allclose(residuals_free, 0.0, atol=1e-4)

    def test_fd_decoder_output_shape(self, formfinder_model, sample_x, bezier_structure):
        """Decoder output should be a flat array of size num_vertices * 3."""
        x_hat = formfinder_model(sample_x, bezier_structure)
        num_vertices = bezier_structure.num_vertices
        assert x_hat.shape == (num_vertices * 3,)


# ---------------------------------------------------------------------------
# AutoEncoder
# ---------------------------------------------------------------------------

class TestAutoEncoder:

    def test_autoencoder_forward(self, formfinder_model, sample_x, bezier_structure):
        """Full model produces output with same shape as input."""
        x_hat = formfinder_model(sample_x, bezier_structure)
        assert x_hat.shape == sample_x.shape

    def test_autoencoder_aux_data(self, formfinder_model, sample_x, bezier_structure):
        """With aux_data=True, returns (x_hat, (q, xyz_fixed, loads))."""
        result = formfinder_model(sample_x, bezier_structure, aux_data=True)
        x_hat, (q, xyz_fixed, loads) = result
        assert x_hat.shape == sample_x.shape
        assert q.ndim == 1
        assert xyz_fixed.ndim == 2
        assert loads.ndim == 2

    def test_autoencoder_predict_states(self, formfinder_model, sample_x, bezier_structure):
        """predict_states returns an EquilibriumState and a ParametersState."""
        eq_state, params_state = formfinder_model.predict_states(
            sample_x, bezier_structure
        )
        # EquilibriumState fields
        assert eq_state.xyz is not None
        assert eq_state.residuals is not None
        assert eq_state.forces is not None

        # ParametersState fields
        assert params_state.q is not None
        assert params_state.xyz_fixed is not None
        assert params_state.loads is not None

    def test_autoencoder_backward(self, formfinder_model, sample_x, bezier_structure):
        """Gradients can be computed through the full model."""

        @eqx.filter_grad
        def grad_fn(model, x, structure):
            x_hat = model(x, structure)
            return jnp.sum(x_hat)

        grads = grad_fn(formfinder_model, sample_x, bezier_structure)
        # Check that at least some gradients are nonzero
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
        assert has_nonzero

    def test_autoencoder_vmap(self, formfinder_model, batch_x, bezier_structure):
        """Model can be vmapped over a batch."""
        predict_fn = jax.vmap(formfinder_model, in_axes=(0, None))
        x_hat_batch = predict_fn(batch_x, bezier_structure)
        assert x_hat_batch.shape == batch_x.shape
