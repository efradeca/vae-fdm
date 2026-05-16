"""Tests for neural_fdm.variational — VAE components.

Verifies mathematical correctness of KL divergence, reparameterization
trick, beta annealing, and full VAE+FDM pipeline.

References tested against:
- Kingma & Welling (2014), Eq. 7: KL divergence formula
- Fu et al. (2019): Cyclical beta annealing
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np
import pytest

from neural_fdm.variational import (
    VariationalAutoEncoder,
    VariationalMLPEncoder,
    compute_beta_schedule,
    compute_kl_divergence,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vae_encoder(prng_key):
    """Small variational encoder for testing."""
    edges_signs = -jnp.ones(180)
    return VariationalMLPEncoder(
        edges_signs=edges_signs, q_shift=0.0,
        in_size=300, out_size=180, width_size=32, depth=2,
        key=prng_key,
    )


@pytest.fixture
def vae_model(bezier_config, bezier_structure, prng_key):
    """Full VAE model with FDM decoder for integration tests."""
    from neural_fdm.builders import (
        build_data_generator,
        build_fd_decoder,
        build_mesh_from_generator,
        calculate_edges_stress_signs,
    )
    gen = build_data_generator(bezier_config)
    mesh = build_mesh_from_generator(bezier_config, gen)
    edges_signs = calculate_edges_stress_signs(mesh)

    k1, k2 = jrn.split(prng_key)
    num_v = mesh.number_of_vertices()
    num_e = mesh.number_of_edges()

    encoder = VariationalMLPEncoder(
        edges_signs=edges_signs, q_shift=0.0,
        in_size=num_v * 3, out_size=num_e,
        width_size=32, depth=2, key=k1,
    )
    decoder = build_fd_decoder(mesh, bezier_config["fdm"])
    return VariationalAutoEncoder(encoder, decoder), gen


# =============================================================================
# KL Divergence Tests
# =============================================================================

class TestKLDivergence:
    """Verify KL(N(mu,sigma) || N(0,I)) against analytical formulas."""

    def test_kl_zero_at_prior(self):
        """KL(N(0,I) || N(0,I)) = 0 exactly (Kingma & Welling 2014, Eq. 7)."""
        mu = jnp.zeros(10)
        log_sigma = jnp.zeros(10)
        kl = compute_kl_divergence(mu, log_sigma)
        assert abs(float(kl)) < 1e-6

    def test_kl_positive(self):
        """KL divergence is always non-negative (Gibbs' inequality)."""
        key = jrn.PRNGKey(42)
        for _ in range(10):
            key, k1, k2 = jrn.split(key, 3)
            mu = jrn.normal(k1, (50,))
            log_sigma = jrn.normal(k2, (50,)) * 0.5
            kl = compute_kl_divergence(mu, log_sigma)
            assert float(kl) >= -1e-6, f"KL should be >= 0, got {float(kl)}"

    def test_kl_known_value(self):
        """KL(N(1,I) || N(0,I)) = D/2 where D is dimensionality.

        From Eq. 7: KL = -0.5 * sum(1 + 0 - 1 - 1) = -0.5 * D * (-1) = D/2
        """
        D = 20
        mu = jnp.ones(D)
        log_sigma = jnp.zeros(D)  # sigma = 1
        kl = compute_kl_divergence(mu, log_sigma)
        expected = D / 2.0
        assert abs(float(kl) - expected) < 1e-4, f"Expected {expected}, got {float(kl)}"

    def test_kl_scales_with_sigma(self):
        """Larger sigma deviation from 1 increases KL."""
        mu = jnp.zeros(10)
        kl_small = compute_kl_divergence(mu, jnp.full(10, -0.1))
        kl_large = compute_kl_divergence(mu, jnp.full(10, 1.0))
        assert float(kl_large) > float(kl_small)

    def test_kl_batched(self):
        """KL works with batched inputs (B, D)."""
        mu = jnp.zeros((5, 10))
        log_sigma = jnp.zeros((5, 10))
        kl = compute_kl_divergence(mu, log_sigma)
        assert abs(float(kl)) < 1e-6


# =============================================================================
# Beta Schedule Tests
# =============================================================================

class TestBetaSchedule:
    """Verify cyclical annealing (Fu et al. 2019)."""

    def test_starts_at_zero(self):
        """Beta = 0 at step 0."""
        assert compute_beta_schedule(0, beta_max=1.0, cycle_length=100) == 0.0

    def test_reaches_max(self):
        """Beta = beta_max after warmup period."""
        beta = compute_beta_schedule(50, beta_max=1.0, cycle_length=100, warmup_ratio=0.5)
        assert abs(beta - 1.0) < 1e-6

    def test_cyclical_reset(self):
        """Beta resets to 0 at start of new cycle."""
        beta = compute_beta_schedule(100, beta_max=1.0, cycle_length=100)
        assert beta == 0.0

    def test_custom_beta_max(self):
        """Beta never exceeds beta_max."""
        for step in range(200):
            beta = compute_beta_schedule(step, beta_max=0.5, cycle_length=100)
            assert beta <= 0.5 + 1e-6

    def test_linear_warmup(self):
        """Beta increases linearly during warmup."""
        b1 = compute_beta_schedule(25, beta_max=1.0, cycle_length=100, warmup_ratio=0.5)
        b2 = compute_beta_schedule(50, beta_max=1.0, cycle_length=100, warmup_ratio=0.5)
        assert abs(b1 - 0.5) < 1e-6
        assert abs(b2 - 1.0) < 1e-6


# =============================================================================
# Variational Encoder Tests
# =============================================================================

class TestVariationalMLPEncoder:
    """Verify encoder architecture and reparameterization."""

    def test_output_shape(self, vae_encoder, prng_key):
        x = jnp.ones(300)
        q, mu, log_sigma = vae_encoder(x, key=prng_key)
        assert q.shape == (180,)
        assert mu.shape == (180,)
        assert log_sigma.shape == (180,)

    def test_sign_constraint(self, vae_encoder, prng_key):
        """All q should be <= 0 (compression) since edges_signs = -1."""
        x = jnp.ones(300)
        q, _, _ = vae_encoder(x, key=prng_key)
        assert bool(jnp.all(q <= 0.001)), f"q max = {float(q.max())}"

    def test_deterministic_when_no_key(self, vae_encoder):
        """Without key, encoder returns MAP estimate (z = mu)."""
        x = jnp.ones(300)
        q1, mu1, _ = vae_encoder(x, key=None)
        q2, mu2, _ = vae_encoder(x, key=None)
        np.testing.assert_allclose(q1, q2, atol=1e-6)

    def test_stochastic_with_different_keys(self, vae_encoder):
        """Different keys produce different samples."""
        x = jnp.ones(300)
        q1, _, _ = vae_encoder(x, key=jrn.PRNGKey(1))
        q2, _, _ = vae_encoder(x, key=jrn.PRNGKey(2))
        assert float(jnp.sum(jnp.abs(q1 - q2))) > 0.01

    def test_same_key_same_output(self, vae_encoder):
        """Same key must produce identical results (reproducibility)."""
        x = jnp.ones(300)
        k = jrn.PRNGKey(42)
        q1, mu1, ls1 = vae_encoder(x, key=k)
        q2, mu2, ls2 = vae_encoder(x, key=k)
        np.testing.assert_allclose(q1, q2, atol=1e-6)

    def test_gradient_flows(self, vae_encoder):
        """Gradients flow through reparameterization trick."""
        x = jnp.ones(300)
        k = jrn.PRNGKey(42)

        def loss(enc):
            q, mu, ls = enc(x, key=k)
            return jnp.sum(q**2) + compute_kl_divergence(mu, ls)

        grads = eqx.filter_grad(loss)(vae_encoder)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        has_grad = any(
            float(jnp.sum(jnp.abs(g))) > 0
            for g in grad_leaves if hasattr(g, "shape")
        )
        assert has_grad, "No gradients detected"

    def test_jit_compatible(self, vae_encoder):
        """Encoder can be JIT-compiled via eqx.filter_jit."""
        x = jnp.ones(300)
        k = jrn.PRNGKey(42)

        @eqx.filter_jit
        def predict(enc, x, k):
            return enc(x, key=k)

        q, mu, ls = predict(vae_encoder, x, k)
        assert q.shape == (180,)


# =============================================================================
# Variational AutoEncoder Tests
# =============================================================================

class TestVariationalAutoEncoder:
    """Verify full VAE + FDM pipeline."""

    def test_forward_deterministic(self, vae_model, bezier_structure, prng_key):
        """Deterministic forward pass produces correct shape."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hat = model(xyz, bezier_structure, aux_data=False, key=None)
        assert x_hat.shape == xyz.shape

    def test_forward_stochastic(self, vae_model, bezier_structure, prng_key):
        """Stochastic forward pass produces correct shape."""
        model, gen = vae_model
        xyz = gen(prng_key)
        k1, k2 = jrn.split(prng_key)
        x_hat = model(xyz, bezier_structure, aux_data=False, key=k2)
        assert x_hat.shape == xyz.shape

    def test_aux_data_format(self, vae_model, bezier_structure, prng_key):
        """aux_data returns ((q, xyz_fixed, loads), mu, log_sigma)."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hat, (params, mu, log_sigma) = model(xyz, bezier_structure, True, key=prng_key)
        q, xyz_fixed, loads = params
        assert q.shape[0] == bezier_structure.num_edges
        assert mu.shape == q.shape
        assert log_sigma.shape == q.shape

    def test_sample_diversity(self, vae_model, bezier_structure, prng_key):
        """sample() generates diverse shapes."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hats, qs = model.sample(xyz, bezier_structure, prng_key, num_samples=5)
        assert x_hats.shape[0] == 5
        # Check diversity: not all identical
        diffs = jnp.sum(jnp.abs(x_hats[0] - x_hats[1]))
        assert float(diffs) > 0.01, "Samples should be diverse"

    def test_all_samples_finite(self, vae_model, bezier_structure, prng_key):
        """All sampled shapes must be finite (no NaN/Inf)."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hats, qs = model.sample(xyz, bezier_structure, prng_key, num_samples=5)
        assert bool(jnp.all(jnp.isfinite(x_hats)))
        assert bool(jnp.all(jnp.isfinite(qs)))

    def test_full_gradient(self, vae_model, bezier_structure, prng_key):
        """Gradients flow through full VAE + FDM pipeline."""
        model, gen = vae_model
        xyz = gen(prng_key)

        def loss(mdl):
            x_hat, (params, mu, ls) = mdl(xyz, bezier_structure, True, key=prng_key)
            return jnp.sum((x_hat - xyz)**2) + compute_kl_divergence(mu, ls)

        grads = eqx.filter_grad(loss)(model)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        has_grad = any(
            float(jnp.sum(jnp.abs(g))) > 0
            for g in grad_leaves if hasattr(g, "shape")
        )
        assert has_grad, "No gradients through VAE+FDM"

    def test_predict_states_interface(self, vae_model, bezier_structure, prng_key):
        """predict_states matches AutoEncoder interface for visualization."""
        model, gen = vae_model
        xyz = gen(prng_key)
        eq_state, fd_params = model.predict_states(xyz, bezier_structure)
        assert hasattr(eq_state, "xyz")
        assert hasattr(eq_state, "forces")
        assert hasattr(eq_state, "residuals")
