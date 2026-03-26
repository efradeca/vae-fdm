"""Tests for VAE loss pipeline integration.

Verifies KL divergence consistency, diversity metrics, and gradient flow.
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
    compute_diversity_metrics,
    compute_kl_divergence,
)


@pytest.fixture
def vae_model(bezier_config, bezier_structure, prng_key):
    """Complete VAE model for testing."""
    from neural_fdm.builders import (
        build_data_generator,
        build_fd_decoder,
        build_mesh_from_generator,
        calculate_edges_stress_signs,
    )
    gen = build_data_generator(bezier_config)
    mesh = build_mesh_from_generator(bezier_config, gen)
    signs = calculate_edges_stress_signs(mesh)
    k1, k2 = jrn.split(prng_key)
    nv, ne = mesh.number_of_vertices(), mesh.number_of_edges()

    enc = VariationalMLPEncoder(
        edges_signs=signs, q_shift=0.0,
        in_size=nv * 3, out_size=ne,
        width_size=32, depth=2, key=k1)
    dec = build_fd_decoder(mesh, bezier_config["fdm"])
    return VariationalAutoEncoder(enc, dec), gen


class TestKLConsistency:
    """Verify KL divergence is consistent between single and batch."""

    def test_kl_single_vs_batch_identical(self):
        """Single sample and batch of identical samples give same KL."""
        key = jrn.PRNGKey(42)
        mu = jrn.normal(key, (180,))
        ls = jrn.normal(key, (180,)) * 0.5

        kl_single = compute_kl_divergence(mu, ls)
        kl_batch = compute_kl_divergence(
            jnp.tile(mu, (8, 1)), jnp.tile(ls, (8, 1)))

        np.testing.assert_allclose(kl_single, kl_batch, atol=1e-4,
            err_msg="KL must be scale-consistent between single and batch")

    def test_kl_batch_mean_equals_manual(self):
        """Batch KL = mean of individual KLs."""
        key = jrn.PRNGKey(99)
        k1, k2, k3, k4 = jrn.split(key, 4)
        mus = jnp.stack([jrn.normal(k1, (50,)), jrn.normal(k2, (50,))])
        lss = jnp.stack([jrn.normal(k3, (50,)) * 0.3, jrn.normal(k4, (50,)) * 0.3])

        kl_batch = compute_kl_divergence(mus, lss)
        kl_0 = compute_kl_divergence(mus[0], lss[0])
        kl_1 = compute_kl_divergence(mus[1], lss[1])
        kl_manual = (kl_0 + kl_1) / 2.0

        np.testing.assert_allclose(kl_batch, kl_manual, atol=1e-4)


class TestVAEGradientFlow:
    """Verify gradients flow through complete VAE+FDM pipeline."""

    def test_grad_through_vae_fmd(self, vae_model, bezier_structure, prng_key):
        """End-to-end gradient: loss -> FDM decoder -> reparameterization -> encoder."""
        model, gen = vae_model
        xyz = gen(prng_key)

        def loss_fn(mdl):
            x_hat, (params, mu, ls) = mdl(xyz, bezier_structure, True, key=prng_key)
            shape_loss = jnp.sum(jnp.abs(x_hat - xyz))
            kl = compute_kl_divergence(mu, ls)
            return shape_loss + 0.5 * kl

        grads = eqx.filter_grad(loss_fn)(model)
        leaves = jax.tree_util.tree_leaves(grads)
        n_grad = sum(1 for g in leaves if hasattr(g, 'shape') and float(jnp.sum(jnp.abs(g))) > 0)
        assert n_grad > 0, "No gradients detected through VAE+FDM"

    def test_grad_encoder_only(self, vae_model, prng_key):
        """Gradients flow through encoder alone."""
        model, gen = vae_model
        xyz = gen(prng_key)

        def loss_fn(enc):
            q, mu, ls = enc(xyz, key=prng_key)
            return jnp.sum(q**2) + compute_kl_divergence(mu, ls)

        grads = eqx.filter_grad(loss_fn)(model.encoder)
        leaves = jax.tree_util.tree_leaves(grads)
        n_grad = sum(1 for g in leaves if hasattr(g, 'shape') and float(jnp.sum(jnp.abs(g))) > 0)
        assert n_grad > 0


class TestDiversityMetrics:
    """Verify solution multiplicity quantification."""

    def test_diversity_positive_for_stochastic(self, vae_model, bezier_structure, prng_key):
        """Stochastic samples should be diverse (pairwise distance > 0)."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hats, qs = model.sample(xyz, bezier_structure, prng_key, num_samples=5)
        metrics = compute_diversity_metrics(x_hats, qs)

        assert metrics["n_samples"] == 5
        assert metrics["q_std_mean"] > 0, "q should vary across samples"
        assert metrics["shape_pairwise_L1_mean"] > 0, "shapes should differ"

    def test_diversity_zero_for_deterministic(self, vae_model, bezier_structure, prng_key):
        """Deterministic mode (same key) should give zero diversity."""
        model, gen = vae_model
        xyz = gen(prng_key)

        # Same key N times = same sample
        same_key = jrn.PRNGKey(0)
        x1, _ = model.sample(xyz, bezier_structure, same_key, num_samples=1)
        x2, _ = model.sample(xyz, bezier_structure, same_key, num_samples=1)
        np.testing.assert_allclose(x1, x2, atol=1e-6)

    def test_all_samples_finite(self, vae_model, bezier_structure, prng_key):
        """Every sampled shape must be finite."""
        model, gen = vae_model
        xyz = gen(prng_key)
        x_hats, qs = model.sample(xyz, bezier_structure, prng_key, num_samples=10)
        assert bool(jnp.all(jnp.isfinite(x_hats)))
        assert bool(jnp.all(jnp.isfinite(qs)))
