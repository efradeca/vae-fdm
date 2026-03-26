from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtu
from jax import vmap
from tqdm import tqdm

from neural_fdm.models import AutoEncoderPiggy


def train_step_piggy(model, structure, optimizer, generator, opt_state, *, loss_fn, batch_size, key):
    """
    Update the parameters of an autoencoder piggy model on a batch of data for one step.

    Parameters
    ----------
    model: `eqx.Module`
        The model to train.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    optimizer: `optax.GradientTransformation`
        The optimizer to use for training.
    generator: `PointGenerator`
        The data generator.
    opt_state: `optax.GradientTransformationExtraArgs`
        The current optimizer state.
    loss_fn: `Callable`
        The loss function.
    batch_size: `int`
        The number of samples to generate in each batch.
    key: `jax.random.PRNGKey`
        The random key.

    Returns
    -------
    loss_vals: `dict` of `float`
        The values of the loss terms.
    model: `eqx.Module`
        The updated model.
    opt_state: `optax.GradientTransformationExtraArgs`
        The updated optimizer state.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates for main
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads_main = val_grad_fn(
        model,
        structure,
        x,
        True,
        False
    )

    # calculate updates for piggy
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads_piggy = val_grad_fn(
        model,
        structure,
        x,
        True,
        True
    )

    # combine gradients
    grads = jtu.tree_map(lambda x, y: x + y, grads_main, grads_piggy)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_step(model, structure, optimizer, generator, opt_state, *, loss_fn, batch_size, key):
    """
    Update the parameters of an autoencoder model on a batch of data for one step.

    Parameters
    ----------
    model: `eqx.Module`
        The model to train.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    optimizer: `optax.GradientTransformation`
        The optimizer to use for training.
    generator: `PointGenerator`
        The data generator.
    opt_state: `optax.GradientTransformationExtraArgs`
        The current optimizer state.
    loss_fn: `Callable`
        The loss function.
    batch_size: `int`
        The number of samples to generate in each batch.
    key: `jax.random.PRNGKey`
        The random key.

    Returns
    -------
    loss_vals: `dict` of `float`
        The values of the loss terms.
    model: `eqx.Module`
        The updated model.
    opt_state: `optax.GradientTransformationExtraArgs`
        The updated optimizer state.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads = val_grad_fn(model, structure, x, aux_data=True)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_step_vae(model, structure, optimizer, generator, opt_state, *,
                   loss_fn, batch_size, key, beta):
    """Training step for VAE models with reparameterization sampling.

    Differs from train_step in two ways:
    1. Passes a PRNG key through the loss function for epsilon sampling.
    2. Accepts beta as a traced JAX float (not Python float) to avoid
       JIT recompilation at each step.

    Parameters
    ----------
    beta : jnp.float32
        Current KL weight. Must be a JAX array (traced), not a Python float,
        to prevent JIT recompilation at every training step.

    References
    ----------
    Kingma & Welling (2014): Reparameterization trick requires PRNG in forward pass.
    Fu et al. (2019): Beta varies per step via cyclical annealing.
    """
    # Split key: one for data generation, one for VAE sampling
    data_key, model_key = jrn.split(key)

    # Sample fresh data
    keys = jrn.split(data_key, batch_size)
    x = vmap(generator)(keys)

    # Create loss wrapper that includes the VAE key and beta
    def vae_loss_wrapper(model, structure, x, aux_data=True):
        from neural_fdm.losses import compute_loss_vae
        # Inject beta into loss_params dynamically
        _loss_params = dict(loss_fn.keywords.get("loss_params", {}))
        _loss_params.setdefault("vae", {})
        _loss_params["vae"]["beta"] = beta
        _loss_fn = loss_fn.keywords.get("loss_fn", None)
        return compute_loss_vae(
            model, structure, x, _loss_fn, _loss_params,
            aux_data=aux_data, key=model_key
        )

    val_grad_fn = eqx.filter_value_and_grad(vae_loss_wrapper, has_aux=True)
    (loss, loss_vals), grads = val_grad_fn(model, structure, x, aux_data=True)

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_model(model, structure, optimizer, generator, *, loss_fn, num_steps, batch_size, key, callback=None):
    """
    Train a model over a number of steps.

    Parameters
    ----------
    model: `eqx.Module`
        The model to train.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    optimizer: `optax.GradientTransformation`
        The optimizer to use for training.
    generator: `PointGenerator`
        The data generator.
    loss_fn: `Callable`
        The loss function.
    num_steps: `int`
        The number of steps to train for (number of parameter updates).
    batch_size: `int`
        The number of samples to generate per batch.
    key: `jax.random.PRNGKey`
        The random key.
    callback: `Callable`, optional
        A callback function to call after each step.
        The callback function should take the following arguments:
        - model: `eqx.Module`
        - opt_state: `optax.GradientTransformationExtraArgs`
        - loss_vals: `dict` of `float`
        - step: `int`
    """
    # initial optimization step
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # detect VAE model
    from neural_fdm.variational import VariationalAutoEncoder
    is_vae = isinstance(model, VariationalAutoEncoder)

    # assemble train step
    if is_vae:
        from neural_fdm.variational import compute_beta_schedule
        # Extract VAE config from loss_fn partial keywords
        _lp = loss_fn.keywords.get("loss_params", {})
        vae_cfg = _lp.get("vae", {})
        beta_max = vae_cfg.get("beta_max", 1.0)
        cycle_length = vae_cfg.get("cycle_length", num_steps)
        warmup_ratio = vae_cfg.get("warmup_ratio", 0.5)

        train_step_fn = partial(train_step_vae, loss_fn=loss_fn)
        train_step_fn = eqx.filter_jit(train_step_fn)
    else:
        train_step_fn = train_step
        if isinstance(model, AutoEncoderPiggy):
            train_step_fn = train_step_piggy
        train_step_fn = partial(train_step_fn, loss_fn=loss_fn)
        train_step_fn = eqx.filter_jit(train_step_fn)

    # train
    loss_history = []
    for step in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        if is_vae:
            # Compute beta as JAX array to avoid JIT recompilation
            beta = jnp.float32(compute_beta_schedule(
                step, beta_max, cycle_length, warmup_ratio
            ))
            loss_vals, model, opt_state = train_step_fn(
                model, structure, optimizer, generator, opt_state,
                batch_size=batch_size, key=key, beta=beta,
            )
        else:
            # train step
            loss_vals, model, opt_state = train_step_fn(
                model, structure, optimizer, generator, opt_state,
                batch_size=batch_size, key=key,
            )

        # store loss values
        loss_history.append(loss_vals)

        # callback
        if callback:
            callback(model, opt_state, loss_vals, step)

    return model, loss_history
