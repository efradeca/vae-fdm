"""Variational Autoencoder for diverse structural form-finding.

Implementation of a VAE coupled with a differentiable Force Density
Method (FDM) decoder for generating diverse equilibrium solutions.

The key insight: the FDM decoder is differentiable (via JAX implicit
differentiation), so the reparameterization trick (Kingma & Welling, 2014)
enables end-to-end training. The decoder is NOT modified -- it remains
the exact physics solver, guaranteeing equilibrium for every sample.

Mathematical formulation:

    Encoder:     mu, log_sigma = E_phi(X_hat)
    Sampling:    z = mu + exp(log_sigma) * epsilon,  epsilon ~ N(0, I)
    Mapping:     q = (softplus(z) + tau) * s
    Decoder:     X(q) = K(q)^{-1} P   (FDM equilibrium, unchanged)
    Loss:        L = L_shape(X, X_hat) + beta * KL(q(z|X_hat) || p(z))

Where:
    - KL divergence: Eq. 7 of Kingma & Welling (2014), arXiv:1312.6114
    - Beta annealing: Cyclical schedule per Fu et al. (2019), NAACL
    - Physics guarantee: FDM enforces R(X;q) = 0 by construction

Motivation (from Pastrana et al., ICLR 2025, Section 6.1):
    "the choice of bar stiffnesses for a given structure is not unique
    and it is potentially appealing to present to the designer a diversity
    of possible solutions by reformulating our model in a variational
    setting (Kingma and Welling, 2014)"

References
----------
[1] Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes.
    ICLR 2014. arXiv:1312.6114
[2] Fu, H. et al. (2019). Cyclical Annealing Schedule: A Simple Approach
    to Mitigating KL Vanishing. NAACL 2019.
[3] Higgins, I. et al. (2017). beta-VAE: Learning Basic Visual Concepts
    with a Constrained Variational Framework. ICLR 2017.
[4] Pastrana, R. et al. (2025). Real-Time Design of Architectural Structures
    with Differentiable Mechanics and Neural Networks. ICLR 2025.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray

# =============================================================================
# Variational Encoder
# =============================================================================


class VariationalMLPEncoder(eqx.Module):
    """Variational MLP encoder for structural form-finding.

    Maps target shapes to a diagonal Gaussian distribution in latent space,
    then samples and maps to valid force densities via softplus + sign.

    Architecture:
        x -> [backbone MLP] -> h -> [mu_head]        -> mu
                                  -> [log_sigma_head] -> log_sigma
        z = mu + exp(log_sigma) * epsilon   (reparameterization trick [1])
        q = (softplus(z) + q_shift) * edges_signs

    The backbone shares features between mu and log_sigma heads,
    improving parameter efficiency (standard VAE practice [1]).

    Parameters
    ----------
    backbone : eqx.nn.MLP
        Shared feature extractor (depth-1 hidden layers).
    mu_head : eqx.nn.Linear
        Maps features to mean vector (unconstrained).
    log_sigma_head : eqx.nn.Linear
        Maps features to log-std vector. Bias initialized to -2.0
        to start with small variance (sigma ~ 0.135), preventing
        initial noise from destabilizing FDM [3, Section 4.1].
    edges_signs : Array
        +1 for tension, -1 for compression per edge.
    q_shift : float
        Minimum force density magnitude (tau in paper [4]).
    """

    backbone: eqx.nn.MLP
    mu_head: eqx.nn.Linear
    log_sigma_head: eqx.nn.Linear
    edges_signs: Array
    q_shift: Float
    slice_out: bool
    slice_indices: Array

    def __init__(
        self,
        edges_signs,
        q_shift=0.0,
        slice_out=False,
        slice_indices=None,
        in_size=300,
        out_size=180,
        width_size=256,
        depth=3,
        activation=jax.nn.elu,
        *,
        key,
    ):
        k1, k2, k3 = jrn.split(key, 3)

        # Shared backbone: depth-1 hidden layers
        # Output is width_size features fed to both heads
        self.backbone = eqx.nn.MLP(
            in_size=in_size,
            out_size=width_size,
            width_size=width_size,
            depth=max(depth - 1, 1),
            activation=activation,
            key=k1,
        )

        # Mean head: no activation (unconstrained)
        self.mu_head = eqx.nn.Linear(width_size, out_size, key=k2)

        # Log-sigma head: bias initialized to -2.0 for small initial variance
        # This is critical to prevent the FDM decoder from receiving
        # highly noisy q values at the start of training [3]
        self.log_sigma_head = eqx.nn.Linear(width_size, out_size, key=k3)
        # Override bias initialization
        new_bias = jnp.full((out_size,), -2.0)
        self.log_sigma_head = eqx.tree_at(
            lambda l: l.bias, self.log_sigma_head, new_bias
        )

        self.edges_signs = edges_signs
        self.q_shift = q_shift
        self.slice_out = slice_out if slice_out else False
        self.slice_indices = slice_indices

    def __call__(
        self,
        x: Float[Array, "N3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "E"], Float[Array, "E"], Float[Array, "E"]]:
        """Encode target shape to force density distribution and sample.

        Parameters
        ----------
        x : Array
            Flat target shape (N*3,).
        key : PRNGKey or None
            Random key for sampling. If None, uses deterministic MAP
            estimate z = mu (no sampling).

        Returns
        -------
        q : Array (E,)
            Force densities (physically valid: correct signs and shift).
        mu : Array (E,)
            Mean of approximate posterior q(z|x).
        log_sigma : Array (E,)
            Log standard deviation of approximate posterior.
        """
        # Optional input slicing (same as Encoder, models.py:273-276)
        if self.slice_out:
            x = jnp.reshape(x, (-1, 3))
            x = x[self.slice_indices, :]
            x = jnp.ravel(x)

        # Shared feature extraction
        h = self.backbone(x)

        # Distribution parameters
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)

        # Numerical stability: clamp log_sigma to prevent
        # sigma explosion (>7.4) or exact zero (no KL gradient)
        log_sigma = jnp.clip(log_sigma, -10.0, 2.0)

        # Reparameterization trick (Kingma & Welling 2014, Eq. 4):
        # z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        # This makes the sampling differentiable w.r.t. mu and sigma
        if key is not None:
            epsilon = jrn.normal(key, shape=mu.shape)
            z = mu + jnp.exp(log_sigma) * epsilon
        else:
            # Deterministic mode: MAP estimate (no sampling)
            z = mu

        # Map to valid force densities
        # softplus ensures positivity, then shift and sign are applied
        # (same convention as MLPEncoder, models.py:332)
        q = (jax.nn.softplus(z) + self.q_shift) * self.edges_signs

        return q, mu, log_sigma


# =============================================================================
# Variational Autoencoder
# =============================================================================


class VariationalAutoEncoder(eqx.Module):
    """Variational autoencoder with differentiable FDM decoder.

    Couples a variational encoder with the physics-based FDM decoder.
    The decoder is NOT modified -- equilibrium is guaranteed for every
    sample from the approximate posterior.

    This enables generation of diverse equilibrium solutions from a
    single target shape, addressing the non-uniqueness of force density
    solutions noted in Pastrana et al. (2025), Section 6.1.

    Parameters
    ----------
    encoder : VariationalMLPEncoder
        Variational encoder producing (q, mu, log_sigma).
    decoder : FDDecoder
        Physics-based decoder (unchanged from deterministic model).
    """

    encoder: VariationalMLPEncoder
    decoder: eqx.Module

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(
        self,
        x: Float[Array, "N3"],
        structure,
        aux_data: bool = False,
        *args,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ):
        """Forward pass: encode, sample, decode.

        Parameters
        ----------
        x : Array
            Flat target shape.
        structure : EquilibriumStructure
            Mesh structure.
        aux_data : bool
            If True, return auxiliary data for loss computation.
        key : PRNGKey or None
            Random key for reparameterization sampling.

        Returns
        -------
        x_hat : Array
            Predicted equilibrium shape.
        vae_data : tuple (only when aux_data=True)
            ((q, xyz_fixed, loads), mu, log_sigma)
        """
        q, mu, log_sigma = self.encoder(x, key=key)
        x_hat = self.decoder(q, x, structure, aux_data)

        if aux_data:
            x_hat, params = x_hat  # params = (q, xyz_fixed, loads)
            return x_hat, (params, mu, log_sigma)

        return x_hat

    def encode(self, x, *, key=None):
        """Encode target to distribution parameters."""
        return self.encoder(x, key=key)

    def decode(self, q, *args, **kwargs):
        """Decode force densities to equilibrium shape."""
        return self.decoder(q, *args, **kwargs)

    def sample(
        self,
        x: Float[Array, "N3"],
        structure,
        key: PRNGKeyArray,
        num_samples: int = 10,
    ) -> tuple[Float[Array, "S N3"], Float[Array, "S E"]]:
        """Generate diverse equilibrium shapes from a single target.

        Samples multiple z values from q(z|x) and decodes each through
        the FDM solver. Every sample is guaranteed to be in equilibrium.

        Parameters
        ----------
        x : Array
            Single target shape (flat).
        structure : EquilibriumStructure
            Mesh structure.
        key : PRNGKey
            Random key.
        num_samples : int
            Number of diverse solutions to generate.

        Returns
        -------
        x_hats : Array (num_samples, N*3)
            Diverse equilibrium shapes.
        qs : Array (num_samples, E)
            Corresponding force densities.
        """
        keys = jrn.split(key, num_samples)

        def _sample_one(k):
            q, _, _ = self.encoder(x, key=k)
            x_hat = self.decoder(q, x, structure, False)
            return x_hat, q

        x_hats, qs = jax.vmap(_sample_one)(keys)
        return x_hats, qs

    def predict_states(self, x, structure):
        """Deterministic prediction for visualization.

        Uses MAP estimate (key=None) for compatibility with the
        existing visualization pipeline.
        """
        x_hat, (params, mu, log_sigma) = self(x, structure, True, key=None)
        from neural_fdm.models import build_states
        return build_states(x_hat, params, structure)


# =============================================================================
# KL Divergence
# =============================================================================


def compute_kl_divergence(
    mu: Float[Array, "... D"],
    log_sigma: Float[Array, "... D"],
) -> Float[Array, ""]:
    """KL divergence between diagonal Gaussian and standard normal.

    KL(q(z|x) || p(z)) where:
        q(z|x) = N(mu, diag(sigma^2))
        p(z)   = N(0, I)

    Formula (Kingma & Welling 2014, Appendix B, Eq. 7):
        KL = -0.5 * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
           = -0.5 * sum_j (1 + 2*log_sigma_j - mu_j^2 - exp(2*log_sigma_j))

    Parameters
    ----------
    mu : Array (..., D)
        Mean of approximate posterior.
    log_sigma : Array (..., D)
        Log standard deviation of approximate posterior.

    Returns
    -------
    kl : scalar
        Mean KL divergence over batch.

    References
    ----------
    [1] Kingma & Welling (2014), arXiv:1312.6114, Eq. 7
    """
    kl_per_dim = -0.5 * (
        1.0 + 2.0 * log_sigma - jnp.square(mu) - jnp.exp(2.0 * log_sigma)
    )
    # Sum over latent dimensions (per-sample KL), then mean over batch.
    # Consistent for both single sample (1D) and batch (2D):
    # - 1D: sum reduces to scalar, mean of scalar = scalar
    # - 2D: sum per row, mean over rows
    kl_per_sample = jnp.sum(kl_per_dim, axis=-1)
    return jnp.mean(kl_per_sample)


# =============================================================================
# Beta Annealing Schedule
# =============================================================================


def compute_beta_schedule(
    step: int,
    beta_max: float = 1.0,
    cycle_length: int = 5000,
    warmup_ratio: float = 0.5,
) -> float:
    """Cyclical beta annealing schedule.

    Beta linearly increases from 0 to beta_max during the warmup
    portion of each cycle, then stays at beta_max for the rest.
    This prevents posterior collapse by allowing the encoder to
    first learn a good reconstruction, then gradually enforce
    the prior constraint.

    Schedule per Fu et al. (2019):
        beta(t) = beta_max * min(1, (t mod T) / (T * r))

    where T = cycle_length, r = warmup_ratio.

    Parameters
    ----------
    step : int
        Current training step.
    beta_max : float
        Maximum beta value. beta=1.0 gives standard VAE ELBO.
        beta<1.0 allows reconstruction to dominate (underfitting prior).
        beta>1.0 gives beta-VAE (Higgins et al. 2017) for stronger
        disentanglement.
    cycle_length : int
        Number of steps per annealing cycle.
    warmup_ratio : float
        Fraction of cycle for linear warmup (0 to 1).

    Returns
    -------
    beta : float
        Current beta value in [0, beta_max].

    References
    ----------
    [2] Fu et al. (2019). Cyclical Annealing Schedule. NAACL 2019.
    [3] Higgins et al. (2017). beta-VAE. ICLR 2017.
    """
    position = step % cycle_length
    warmup_length = cycle_length * warmup_ratio  # float division, no rounding
    beta = beta_max * min(1.0, position / max(warmup_length, 1e-10))
    return beta


# =============================================================================
# Solution Multiplicity Metrics
# =============================================================================


def compute_diversity_metrics(
    x_hats: Float[Array, "S N3"],
    qs: Float[Array, "S E"],
) -> dict:
    """Quantify solution multiplicity from VAE samples.

    Given S samples of equilibrium shapes and force densities from the
    same target, computes metrics characterizing the diversity of solutions.

    Quantifies the force density solution multiplicity documented
    qualitatively by Veenendaal & Block (2012) and Adriaenssens et al. (2014).

    Parameters
    ----------
    x_hats : Array (S, N*3)
        S sampled equilibrium shapes from the same target.
    qs : Array (S, E)
        Corresponding force density vectors.

    Returns
    -------
    metrics : dict
        - "n_samples": number of samples
        - "shape_pairwise_L1_mean": mean pairwise L1 distance between shapes
        - "shape_pairwise_L1_std": std of pairwise L1 distances
        - "q_pairwise_L1_mean": mean pairwise L1 distance between q vectors
        - "q_std_per_edge": Array (E,) std of q across samples per edge
        - "q_std_mean": mean of per-edge std (scalar summary)
        - "shape_std_per_node": Array (N,) std of position across samples per node

    References
    ----------
    Veenendaal & Block (2012). "An overview and comparison of structural
    form finding methods." IJSS, 49(26):3741-3753.
    Adriaenssens et al. (2014). Shell Structures for Architecture. Routledge.
    """
    S = x_hats.shape[0]

    # Pairwise L1 distances (shapes)
    shape_dists = []
    q_dists = []
    for i in range(S):
        for j in range(i + 1, S):
            shape_dists.append(float(jnp.sum(jnp.abs(x_hats[i] - x_hats[j]))))
            q_dists.append(float(jnp.sum(jnp.abs(qs[i] - qs[j]))))

    import numpy as np
    shape_dists = np.array(shape_dists)
    q_dists = np.array(q_dists)

    # Per-edge q standard deviation (where is there freedom?)
    q_std = np.std(np.array(qs), axis=0)

    # Per-node shape standard deviation
    x_reshaped = np.array(x_hats).reshape(S, -1, 3)
    shape_std = np.std(np.linalg.norm(x_reshaped, axis=-1), axis=0)

    return {
        "n_samples": S,
        "shape_pairwise_L1_mean": float(np.mean(shape_dists)) if len(shape_dists) > 0 else 0.0,
        "shape_pairwise_L1_std": float(np.std(shape_dists)) if len(shape_dists) > 0 else 0.0,
        "q_pairwise_L1_mean": float(np.mean(q_dists)) if len(q_dists) > 0 else 0.0,
        "q_std_per_edge": q_std,
        "q_std_mean": float(np.mean(q_std)),
        "shape_std_per_node": shape_std,
    }


def compute_variance_per_edge(
    model,
    x_target: Float[Array, "N3"],
    structure,
    key,
    n_samples: int = 50,
) -> Float[Array, "E"]:
    """Map solution freedom to individual structural members.

    Samples n_samples force density vectors from the VAE posterior
    and computes the variance of q per edge. High variance indicates
    the structure has design freedom at that member; low variance
    indicates the member is strongly constrained by the target geometry.

    This enables visualization of "structural design freedom" per member,
    a capability enabled by the variational formulation.

    Parameters
    ----------
    model : VariationalAutoEncoder
        Trained VAE model.
    x_target : Array (N*3,)
        Target shape.
    structure : EquilibriumStructure
        Mesh structure.
    key : PRNGKey
        Random key for sampling.
    n_samples : int
        Number of samples (more = better estimate).

    Returns
    -------
    q_variance : Array (E,)
        Variance of force density per edge across samples.
    """
    _, qs = model.sample(x_target, structure, key, num_samples=n_samples)
    return jnp.var(qs, axis=0)
