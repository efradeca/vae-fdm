import jax
import jax.numpy as jnp
from jax import vmap

from neural_fdm.helpers import vertices_residuals_from_xyz
from neural_fdm.models import AutoEncoderPiggy

# ===============================================================================
# Loss assemblers
# ===============================================================================

def compute_loss(
    model,
    structure,
    x,
    loss_fn,
    loss_params,
    aux_data=False,
    piggy_mode=False
):
    """
    Compute the model loss according to the model type.

    Parameters
    ----------
    model: `eqx.Module`
        The model.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    x: `jax.Array`
        The target shape.
    loss_fn: `Callable`
        The loss function.
    loss_params: `dict`
        The scaling parameters to combine the loss' error terms.
    aux_data: `bool`
        If true, returns auxiliary data.
    piggy_mode: `bool`
        If true, the model is a piggy autoencoder.

    Returns
    -------
    loss: `float` or `tuple`
        The loss. If `aux_data` is `True`, returns a tuple of the loss and the loss terms.
    """
    predict_fn = vmap(model, in_axes=(0, None, None, None))
    x_hat, data_hat = predict_fn(x, structure, True, piggy_mode)

    _loss_fn = _compute_loss
    if isinstance(model, AutoEncoderPiggy):
        _loss_fn = _compute_loss_piggy

    loss = _loss_fn(
        loss_fn,
        loss_params,
        x,
        x_hat,
        data_hat,
        structure,
        aux_data,
        piggy_mode
    )

    return loss


def _compute_loss(
    loss_fn,
    loss_params,
    x,
    x_hat,
    params_hat,
    structure,
    aux_data,
    piggy_mode=False
):
    """
    Compute the model loss of an autoencoder.

    Parameters
    ----------
    loss_fn: `Callable`
        The loss function.
    loss_params: `dict`
        The scaling parameters to combine the loss' error terms.
    x: `jax.Array`
        The target shape.
    x_hat: `jax.Array`
        The predicted shape.
    params_hat: tuple of `jax.Array`
        The predicted force densities, loads, and fixed positions.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    aux_data: `bool`
        If true, returns auxiliary data.
    piggy_mode: `bool`
        If true, the model is a piggy autoencoder.

    Returns
    -------
    loss: `float` or `tuple`
        The loss. If `aux_data` is `True`, returns a tuple of the loss and the loss terms.
    """
    return loss_fn(x, x_hat, params_hat, structure, loss_params, aux_data)


def _compute_loss_piggy(
    loss_fn,
    loss_params,
    x,
    x_data_hat,
    y_data_hat,
    structure,
    aux_data,
    piggy_mode=True,
):
    """
    Compute the loss of a piggy autoencoder.

   Parameters
    ----------
    loss_fn: `Callable`
        The loss function.
    loss_params: `dict`
        The scaling parameters to combine the loss' error terms.
    x: `jax.Array`
        The target shape.
    x_data_hat: `tuple`
        The predicted shape and the predicted parameters.
    y_data_hat: `tuple`
        The predicted shape and the predicted parameters.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    aux_data: `bool`
        If true, returns auxiliary data.
    piggy_mode: `bool`
        If true, the model is a piggy autoencoder.

    Returns
    -------
    loss: `float` or `tuple`
        The loss. If `aux_data` is `True`, returns a tuple of the loss and the loss terms.
    """
    x_hat, x_params_hat = x_data_hat

    if not piggy_mode:
        loss_data = loss_fn(x, x_hat, x_params_hat, structure, loss_params, aux_data)
    else:
        y_hat, y_params_hat = y_data_hat
        loss_data = loss_fn(x_hat, y_hat, y_params_hat, structure, loss_params, aux_data)

    return loss_data


# ===============================================================================
# Task losses
# ===============================================================================

def compute_loss_shell(
    x,
    x_hat,
    params_hat,
    structure,
    loss_params,
    aux_data,
    *args
):
    """
    Compute the loss for the shell task.

    Parameters
    ----------
    x: `jax.Array`
        The target shape.
    x_hat: `jax.Array`
        The predicted shape.
    params_hat: tuple of `jax.Array`
        The predicted force densities, loads, and fixed positions.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    loss_params: `dict`
        The scaling parameters to combine the loss' error terms.
    aux_data: `bool`
        If true, returns auxiliary data.

    Returns
    -------
    loss: `float` or `tuple`
        The loss. If `aux_data` is `True`, returns a tuple of the loss and the loss terms.
    """
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"]
    loss_shape = compute_error_shape_l1(x, x_hat)
    loss_shape = factor_shape * loss_shape

    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual

    loss_terms = {
        "loss": loss,
        "shape error": loss_shape,
        "residual error": loss_residual
    }

    if aux_data:
        return loss, loss_terms

    return loss


def compute_loss_tower(
        x,
        x_hat,
        params_hat,
        structure,
        loss_params,
        aux_data,
        *args
):
    """
    Compute the loss for the tower task.

    Parameters
    ----------
    x: `jax.Array`
        The target shape.
    x_hat: `jax.Array`
        The predicted shape.
    params_hat: tuple of `jax.Array`
        The predicted force densities, loads, and fixed positions.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    loss_params: `dict`
        The scaling parameters to combine the loss' error terms.
    aux_data: `bool`
        If true, returns auxiliary data.

    Returns
    -------
    loss: `float` or `tuple`
        The loss. If `aux_data` is `True`, returns a tuple of the loss and the loss terms.
    """
    # compression ring shape
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"]
    shape_dims = shape_params["dims"]
    levels_compression = shape_params["levels_compression"]

    def slice_xyz_rings(_x, levels):
        return jnp.reshape(_x, shape_dims)[levels, :, :].ravel()

    slice_xyz_vmap = vmap(slice_xyz_rings, in_axes=(0, None))
    xyz_slice = slice_xyz_vmap(x, levels_compression)
    xyz_hat_slice = slice_xyz_vmap(x_hat, levels_compression)
    assert xyz_slice.shape == xyz_hat_slice.shape

    # NOTE: Using L2 norm here because L1 does not work well
    loss_shape = compute_error_shape_l2(xyz_slice, xyz_hat_slice)
    loss_shape = factor_shape * loss_shape

    # tension rings height
    height_params = loss_params["shape"]
    factor_height = height_params["weight"]
    height_dims = height_params["dims"]
    levels_tension = height_params["levels_tension"]

    def slice_z_rings(_x, levels):
        return jnp.reshape(_x, height_dims)[levels, :, 2].ravel()

    slice_z_vmap = vmap(slice_z_rings, in_axes=(0, None))
    z_slice = slice_z_vmap(x, levels_tension)
    z_hat_slice = slice_z_vmap(x_hat, levels_tension)
    assert z_slice.shape == z_hat_slice.shape

    # NOTE: Using L2 norm here because L1 does not work well
    loss_height = compute_error_shape_l2(z_slice, z_hat_slice)
    loss_height = factor_height * loss_height

    # Add the shape and height losses
    loss_shape = loss_shape + loss_height

    # residual
    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    # regularization
    regularization_params = loss_params["regularization"]
    factor_regularization = regularization_params["weight"]
    q = params_hat[0]
    regularization = compute_q_regularization(q)
    regularization = factor_regularization * regularization

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual
    if regularization_params["include"]:
        loss = loss + regularization

    loss_terms = {
        "loss": loss,
        "shape error": loss_shape,
        "residual error": loss_residual,
        "regularization": regularization
    }

    if aux_data:
        return loss, loss_terms

    return loss


# ===============================================================================
# VAE task losses (extends existing losses with KL divergence)
# ===============================================================================


def compute_loss_shell_vae(
    x, x_hat, params_hat_vae, structure, loss_params, aux_data, *args
):
    """Loss for shell task with VAE: reconstruction + beta * KL.

    Wraps compute_loss_shell and adds the KL divergence term from the
    variational encoder. The KL weight (beta) follows the cyclical
    annealing schedule of Fu et al. (NAACL 2019).

    Parameters
    ----------
    params_hat_vae : tuple
        ((q, xyz_fixed, loads), mu, log_sigma) - VAE extended aux_data.
    loss_params : dict
        Must include 'vae' key with 'beta' value.

    References
    ----------
    Kingma & Welling (2014), Eq. 7: KL divergence formula.
    Fu et al. (2019): Cyclical beta annealing.
    """
    from neural_fdm.variational import compute_kl_divergence

    # Unpack VAE-specific data
    params_hat, mu, log_sigma = params_hat_vae

    # Reconstruction loss (delegates to existing shell loss)
    recon_result = compute_loss_shell(
        x, x_hat, params_hat, structure, loss_params, aux_data
    )

    # KL divergence (Kingma & Welling 2014, Eq. 7)
    kl_loss = compute_kl_divergence(mu, log_sigma)

    # Beta-weighted ELBO (Higgins et al. 2017)
    beta = loss_params.get("vae", {}).get("beta", 0.0)

    if aux_data:
        recon_loss, loss_terms = recon_result
        loss = recon_loss + beta * kl_loss
        loss_terms["loss"] = loss
        loss_terms["kl divergence"] = kl_loss
        loss_terms["beta"] = jnp.float32(beta)
        return loss, loss_terms
    else:
        loss = recon_result + beta * kl_loss
        return loss


def compute_loss_vae(
    model, structure, x, loss_fn, loss_params, aux_data=False, *, key=None
):
    """Loss assembler for VAE models.

    Handles PRNG key threading for the reparameterization trick.
    Each sample in the batch gets a different key for epsilon sampling.

    Parameters
    ----------
    model : VariationalAutoEncoder
    structure : EquilibriumStructure
    x : Array (B, N*3)
        Batch of target shapes.
    loss_fn : callable
        Task-specific loss (compute_loss_shell_vae or compute_loss_tower_vae).
    loss_params : dict
    aux_data : bool
    key : PRNGKey or None
        If provided, enables stochastic sampling. If None, uses MAP.
    """
    from jax import vmap

    batch_size = x.shape[0]

    if key is not None:
        keys = jax.random.split(key, batch_size)

        def predict_one(xi, ki):
            return model(xi, structure, True, key=ki)

        predict_fn = vmap(predict_one, in_axes=(0, 0))
        x_hat, data_hat = predict_fn(x, keys)
    else:
        def predict_one_det(xi):
            return model(xi, structure, True, key=None)

        predict_fn = vmap(predict_one_det)
        x_hat, data_hat = predict_fn(x)

    return loss_fn(x, x_hat, data_hat, structure, loss_params, aux_data)


# ===============================================================================
# Shape approximation error
# ===============================================================================

def compute_error_shape_l1(x, x_hat):
    """
    Calculate the L1 shape reconstruction error, averaged over the batch.

    Parameters
    ----------
    x: `jax.Array`
        The target shape.
    x_hat: `jax.Array`
        The predicted shape.

    Returns
    -------
    error: `float`
        The reconstruction error.
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_error_shape_l2(x, x_hat):
    """
    Calculate the L2 shape reconstruction error, averaged over the batch.

    Parameters
    ----------
    x: `jax.Array`
        The target shape.
    x_hat: `jax.Array`
        The predicted shape.

    Returns
    -------
    error: `float`
        The reconstruction error.
    """
    error = jnp.square(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


# ===============================================================================
# Residual error
# ===============================================================================

def compute_error_residual(x_hat, params_hat, structure, indices):
    """
    Calculate the residual error, averaged over the batch. This is the physics loss.

    Parameters
    ----------
    x_hat: `jax.Array`
        The predicted shape.
    params_hat: tuple of `jax.Array`
        The predicted force densities, loads, and fixed positions.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    indices: `jax.Array`
        The indices of the free vertices to calculate the residual at.

    Returns
    -------
    error: `float`
        The residual error.
    """
    def calculate_residuals(_x_hat, _params_hat):
        # NOTE: Not using jnp.linalg.norm because we hitted NaNs.
        q_hat, xyz_fixed, loads = _params_hat
        residual_vectors = vertices_residuals_from_xyz(q_hat, loads, _x_hat, structure)
        residual_vectors_free = jnp.ravel(residual_vectors[indices, :])

        # return jnp.linalg.norm(residual_vectors_free, axis=-1)
        # return jnp.sqrt(jnp.sum(jnp.square(residual_vectors_free), axis=-1))
        return jnp.square(residual_vectors_free)

    residuals = vmap(calculate_residuals)(x_hat, params_hat)
    shape_residuals = jnp.sqrt(jnp.sum(residuals, axis=-1))
    batch_residual = jnp.mean(shape_residuals, axis=-1)

    return batch_residual


# ===============================================================================
# Regularization
# ===============================================================================

def compute_q_regularization(q):
    """
    Calculate variance of the force densities for compression and tension.

    Parameters
    ----------
    q: `jax.Array`
        The force densities.

    Returns
    -------
    result: `float`
        The sum of the two variances.
    """
    sign_q = jnp.sign(q)
    var_q_pos = jnp.var(q, where=sign_q > 0)
    var_q_neg = jnp.var(q, where=sign_q < 0)

    # NOTE: jnp.mean is doing nothing here because the size of the variance arrays is 1
    result = jnp.mean(var_q_pos) + jnp.mean(var_q_neg)

    return result




# ===============================================================================
# Utilities
# ===============================================================================

def print_loss_summary(loss_terms, prefix=None):
    """
    Print a summary of the loss terms.

    Parameters
    ----------
    loss_terms: `dict`
        The loss terms.
    prefix: `str` or `None`, optional
        The prefix to add to the loss terms printed to the console.
    """
    msg_parts = []
    if prefix:
        msg_parts.append(prefix)

    for label, term in loss_terms.items():
        part = f"{label.capitalize()}: {term.item():.4f}"
        msg_parts.append(part)

    msg = "\t".join(msg_parts)
    print(msg)
