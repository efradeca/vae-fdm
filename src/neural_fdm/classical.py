"""Classical baseline: direct optimization of force densities.

Solves the inverse form-finding problem by optimizing force densities q
to minimize shape error, using the same differentiable FDM decoder as
the neural approach. This provides the classical comparison baseline
for evaluating neural encoder performance.

The optimization uses bounded L-BFGS-B (via jaxopt) with box constraints
that respect the compression/tension signs of each edge.

References
----------
[1] Schek, H.J. (1974). The force density method for form finding and
    computation of general networks. CMAME, 3(1):115-134.
[2] Pastrana, R. et al. (2025). Neural FDM. ICLR 2025. Section 4.
"""

from __future__ import annotations

from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrn
from jaxopt import ScipyBoundedMinimize

from neural_fdm.builders import (
    build_fd_decoder_parametrized,
    build_loss_function,
)
from neural_fdm.helpers import compute_l_physics


def _build_bounds(mesh, q0, qmin, qmax):
    """Construct box constraints respecting edge stress signs."""
    bound_low, bound_up = [], []
    for edge in mesh.edges():
        if mesh.edge_attribute(edge, "tag") == "cable":
            bound_low.append(qmin)
            bound_up.append(qmax)
        else:
            bound_low.append(-qmax)
            bound_up.append(-qmin)
    return jnp.array(bound_low), jnp.array(bound_up)


def _init_q(mesh, key, qmin=0.1, qmax=10.0, initial_q=None):
    """Initialize force densities with correct signs."""
    num_edges = mesh.number_of_edges()
    signs = []
    for edge in mesh.edges():
        sign = -1.0
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1.0
        signs.append(sign)
    signs = jnp.array(signs)

    if initial_q is not None:
        return initial_q
    return jrn.uniform(key, shape=(num_edges,), minval=qmin, maxval=qmax) * signs


def solve_classical(
    xyz_target,
    config,
    mesh,
    structure,
    method="L-BFGS-B",
    maxiter=2000,
    tol=1e-6,
    initial_q=None,
    key=None,
    qmin=0.1,
    qmax=10.0,
):
    """Solve form-finding via direct optimization of force densities.

    Parameters
    ----------
    xyz_target : jax.Array
        Target vertex positions (flat, shape N*3).
    config : dict
        YAML configuration (same as used for neural training).
    mesh : FDMesh
        The FDM mesh.
    structure : EquilibriumMeshStructure
        The connectivity structure.
    method : str
        Scipy optimizer method ("L-BFGS-B" or "SLSQP").
    maxiter : int
        Maximum optimization iterations.
    tol : float
        Convergence tolerance.
    initial_q : jax.Array, optional
        Initial force densities (warm-start). If None, random init.
    key : PRNGKey, optional
        Random key for q initialization. Required if initial_q is None.
    qmin : float
        Minimum absolute q value for bounds and init.
    qmax : float
        Maximum absolute q value for bounds and init.

    Returns
    -------
    result : dict
        q_opt, xyz_opt, l_shape, l_physics, loss_total,
        success, n_iters, runtime_ms
    """
    if key is None and initial_q is None:
        key = jrn.PRNGKey(0)

    fd_params = config["fdm"]
    q0 = _init_q(mesh, key, qmin, qmax, initial_q)

    # Build parametrized decoder (q is the only trainable parameter)
    decoder = build_fd_decoder_parametrized(q0, mesh, fd_params)

    # Build loss function (same as neural training)
    from neural_fdm.builders import build_data_generator
    generator = build_data_generator(config)
    compute_loss = build_loss_function(config, generator)

    # Split: only q is trainable, everything else is static
    filter_spec = jax.tree_util.tree_map(lambda _: False, decoder)
    filter_spec = eqx.tree_at(lambda t: t.q, filter_spec, replace=True)
    diff_model, static_model = eqx.partition(decoder, filter_spec)

    # Loss wrapper for jaxopt — only q is differentiated
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_fn(diff_model, xyz):
        model = eqx.combine(diff_model, static_model)
        return compute_loss(model, structure, xyz, aux_data=False)

    # Bounds (only for q)
    bound_low, bound_up = _build_bounds(mesh, q0, qmin, qmax)
    bl_tree = eqx.tree_at(lambda t: t.q, diff_model, replace=bound_low)
    bu_tree = eqx.tree_at(lambda t: t.q, diff_model, replace=bound_up)
    bounds = (bl_tree, bu_tree)

    # Optimizer
    opt = ScipyBoundedMinimize(
        fun=loss_fn,
        method=method,
        jit=True,
        tol=tol,
        maxiter=maxiter,
        options={"disp": False},
        value_and_grad=True,
    )

    # JIT warmup
    xyz_in = xyz_target[None, :]
    _ = loss_fn(diff_model, xyz_in)

    # Solve
    t0 = perf_counter()
    diff_opt, opt_res = opt.run(diff_model, bounds, xyz_in)
    runtime_ms = (perf_counter() - t0) * 1000

    # Extract results
    model_opt = eqx.combine(diff_opt, static_model)
    x_hat, (q_opt, xyz_fixed, loads) = model_opt(
        xyz_target, structure, aux_data=True
    )

    # Metrics (same as neural evaluation)
    xyz_pred = jnp.reshape(x_hat, (-1, 3))
    xyz_tgt = jnp.reshape(xyz_target, (-1, 3))
    l_shape = float(jnp.sum(jnp.abs(xyz_pred - xyz_tgt)))
    free_indices = sorted(list(mesh.vertices_free()))
    l_physics = compute_l_physics(x_hat, q_opt, loads, structure, free_indices)

    # Check for near-singular q (NaN/Inf in result indicates the FDM
    # linear solve degenerated). Warn but still return so callers can
    # see the partial result.
    success_flag = bool(opt_res.success) if hasattr(opt_res, "success") else True
    if not jnp.all(jnp.isfinite(q_opt)) or not jnp.isfinite(l_shape):
        import warnings
        warnings.warn(
            "classical FDM solver produced non-finite q/l_shape; the equilibrium "
            "matrix likely became near-singular. Marking success=False.",
            RuntimeWarning,
        )
        success_flag = False

    return {
        "q_opt": q_opt,
        "xyz_opt": xyz_pred,
        "l_shape": l_shape,
        "l_physics": l_physics,
        "loss_total": float(opt_res.fun_val) if hasattr(opt_res, "fun_val") else l_shape,
        "success": success_flag,
        "n_iters": int(opt_res.iter_num) if hasattr(opt_res, "iter_num") else -1,
        "runtime_ms": runtime_ms,
    }
