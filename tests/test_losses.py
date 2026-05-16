"""Tests for neural_fdm.losses — loss computation."""


import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from neural_fdm.builders import build_loss_function, build_neural_model
from neural_fdm.losses import compute_error_shape_l1, compute_loss_shell

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def formfinder_model(bezier_config, bezier_generator, prng_key):
    """Build a small formfinder model for testing."""
    return build_neural_model("formfinder", bezier_config, bezier_generator, prng_key)


@pytest.fixture
def loss_fn(bezier_config, bezier_generator):
    """Build the loss function for the bezier task."""
    return build_loss_function(bezier_config, bezier_generator)


@pytest.fixture
def sample_batch(bezier_generator, prng_key):
    """Generate a small batch of samples."""
    keys = jrn.split(prng_key, 2)
    return jax.vmap(bezier_generator)(keys)


# ---------------------------------------------------------------------------
# Shape loss (L1)
# ---------------------------------------------------------------------------

class TestShapeLoss:

    def test_shape_loss_zero_on_identical(self):
        """L_shape should be 0 when x == x_hat."""
        x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
        loss = compute_error_shape_l1(x, x)
        assert jnp.allclose(loss, 0.0, atol=1e-7)

    def test_shape_loss_positive(self):
        """L_shape should be > 0 when x != x_hat."""
        x = jnp.array([[1.0, 2.0, 3.0]])
        x_hat = jnp.array([[1.1, 2.2, 3.3]])
        loss = compute_error_shape_l1(x, x_hat)
        assert loss > 0.0

    def test_shape_loss_symmetric(self):
        """L_shape(x, x_hat) == L_shape(x_hat, x)."""
        x = jnp.array([[1.0, 2.0, 3.0]])
        x_hat = jnp.array([[4.0, 5.0, 6.0]])
        loss_a = compute_error_shape_l1(x, x_hat)
        loss_b = compute_error_shape_l1(x_hat, x)
        assert jnp.allclose(loss_a, loss_b, atol=1e-7)


# ---------------------------------------------------------------------------
# Residual loss for formfinder
# ---------------------------------------------------------------------------

class TestResidualLoss:

    def test_residual_loss_formfinder_zero(
        self, formfinder_model, bezier_structure, sample_batch
    ):
        """For a formfinder model, the residual loss should be 0 (physics decoder satisfies equilibrium)."""
        loss_params = {
            "shape": {"include": False, "weight": 0.0},
            "residual": {"include": True, "weight": 1.0},
        }

        predict_fn = jax.vmap(formfinder_model, in_axes=(0, None, None))
        x_hat, params_hat = predict_fn(sample_batch, bezier_structure, True)

        loss = compute_loss_shell(
            sample_batch, x_hat, params_hat, bezier_structure, loss_params, False
        )
        # Residual should be near zero for the physics-based decoder
        assert loss < 1e-3


# ---------------------------------------------------------------------------
# compute_loss integration
# ---------------------------------------------------------------------------

class TestComputeLoss:

    def test_compute_loss_returns_dict(
        self, formfinder_model, bezier_structure, sample_batch, loss_fn
    ):
        """compute_loss with aux_data returns (loss, dict) with expected keys."""
        loss, loss_terms = loss_fn(
            formfinder_model, bezier_structure, sample_batch, aux_data=True
        )

        assert isinstance(loss_terms, dict)
        assert "loss" in loss_terms
        assert "shape error" in loss_terms
        assert "residual error" in loss_terms

    def test_compute_loss_scalar(
        self, formfinder_model, bezier_structure, sample_batch, loss_fn
    ):
        """compute_loss without aux_data returns a scalar."""
        loss = loss_fn(
            formfinder_model, bezier_structure, sample_batch, aux_data=False
        )
        assert loss.ndim == 0

    def test_compute_loss_nonnegative(
        self, formfinder_model, bezier_structure, sample_batch, loss_fn
    ):
        """Loss should be non-negative."""
        loss = loss_fn(
            formfinder_model, bezier_structure, sample_batch, aux_data=False
        )
        assert loss >= 0.0
