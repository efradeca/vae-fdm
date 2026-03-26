import equinox as eqx
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax_fdm.equilibrium import EquilibriumModel
from jaxtyping import Array, Bool, Float

from neural_fdm.helpers import (
    calculate_area_loads,
    calculate_equilibrium_state,
    calculate_fd_params_state,
)

# ===============================================================================
# Autoencoders
# ===============================================================================

class AutoEncoder(eqx.Module):
    """
    A model that pipes an encoder to a decoder.

    Parameters
    ----------
    encoder: `eqx.Module`
        The encoder.
    decoder: `eqx.Module`
        The decoder.
    """
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x, structure, aux_data=False, *args, **kwargs):
        """
        Predict a shape that approximates the target shape.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.
        aux_data: `bool`, optional
            Whether to return auxiliary data. The auxiliary data is a tuple of the
            force density parameters, the fixed node positions, and the applied loads.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        data: `tuple` of `jax.Array`
            The auxiliary data if `aux_data` is `True`.
        """
        # NOTE: x must be a flat vector
        # GNN encoders need structure for edge connectivity
        from neural_fdm.gnn import GNNEncoder
        if isinstance(self.encoder, GNNEncoder):
            q = self.encoder(x, structure=structure)
        else:
            q = self.encoder(x)
        x_hat = self.decoder(q, x, structure, aux_data)

        return x_hat

    def encode(self, x):
        """
        Generate the latent representation of a target shape.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.

        Returns
        -------
        q: `jax.Array`
            The latent representation.
        """
        return self.encoder(x)

    def decode(self, q, *args, **kwargs):
        """
        Map a latent representation back to shape space.

        Parameters
        ----------
        q: `jax.Array`
            The latent representation.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        return self.decoder(q, *args, **kwargs)

    def predict_states(self, x, structure):
        """
        Predict equilibrium and parameter states for visualization.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        eq_state: `jax_fdm.EquilibriumState`
            The current equilibrium state of the structure.
        fd_params_state: `jax_fdm.EquilibriumParametersState`
            The current state of simulation parameters.
        """
        # Predict shape
        x_hat, params = self(x, structure, True)

        return build_states(x_hat, params, structure)


class AutoEncoderPiggy(AutoEncoder):
    """
    An autoencoder with a piggybacking decoder.

    Parameters
    ----------
    encoder: `eqx.Module`
        The encoder.
    decoder: `eqx.Module`
        The decoder.
    decoder_piggy: `eqx.Module`
        The piggybacking decoder.
    """
    decoder_piggy: eqx.Module

    def __init__(self, encoder, decoder, decoder_piggy):
        super().__init__(encoder, decoder)
        self.decoder_piggy = decoder_piggy

    def __call__(self, x, structure, aux_data=False, piggy_mode=True):
        """
        Predict a shape that approximates the target shape.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.
        aux_data: `bool`, optional
            Whether to return auxiliary data. The auxiliary data is a tuple of the
            force density parameters, the fixed node positions, and the applied loads.
        piggy_mode: `bool`, optional
            Whether to use the piggybacking decoder. If `True`, gradients are not backpropagated
            from the piggybacking decoder into the encoder.

        Returns
        -------
        x_hat: `jax.Array` or `tuple` of `jax.Array`
            The predicted shape. If `aux_data` is `True`, this is a tuple of the
            predicted shape and the auxiliary data.
        y_hat: `jax.Array` or `tuple` of `jax.Array`
            The predicted shape from the piggybacking decoder. If `aux_data` is `True`,
            this is a tuple of the predicted shape and the auxiliary data from
            the piggybacking decoder.
        """
        q = self.encoder(x)
        x_hat = self.decoder(q, x, structure, aux_data)

        if piggy_mode:
            q = stop_gradient(q)
            x_hat = stop_gradient(x_hat)

        y_hat = self.decoder_piggy(q, x, structure, aux_data)

        return x_hat, y_hat

    def decode(self, q, *args, **kwargs):
        """
        Map a latent representation back to shape space.

        Parameters
        ----------
        q: `jax.Array`
            The latent representation.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        return self.decoder_piggy(q, *args, **kwargs)

    def predict_states(self, x, structure):
        """
        Predict equilibrium and parameter states for visualization.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        eq_state: `jax_fdm.EquilibriumState`
            The current equilibrium state of the structure.
        fd_params_state: `jax_fdm.EquilibriumParametersState`
            The current state of simulation parameters.
        """
        # Predict shape
        _, pred_piggy = self(x, structure, True)
        x_hat, params = pred_piggy

        return build_states(x_hat, params, structure)


# ===============================================================================
# Encoders
# ===============================================================================

class Encoder(eqx.Module):
    """
    An encoder.

    Parameters
    ----------
    edges_signs: `jax.Array`
        An array of +1s to denote tension and -1s to denote compression on the edges.
    q_shift: `float`, optional
        The minimum value of the latent representation.
    slice_out: `bool`, optional
        Whether to slice the output of the encoder to learn a mapping only
        w.r.t. a slice of the target shape.
    slice_indices: `jax.Array`, optional
        The indices of the points to slice from the target shape.
    """
    edges_signs: Array
    q_shift: Float
    slice_out: Bool
    slice_indices: Array

    def __init__(
            self,
            edges_signs,
            q_shift=0.0,
            slice_out=False,
            slice_indices=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.edges_signs = edges_signs
        self.q_shift = q_shift
        self.slice_out = slice_out
        self.slice_indices = slice_indices

    def __call__(self, x):
        """
        Map a target shape to a latent representation.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.

        Returns
        -------
        q: `jax.Array`
            The latent representation.
        """
        if self.slice_out:
            x = jnp.reshape(x, (-1, 3))
            x = x[self.slice_indices, :]
            x = jnp.ravel(x)

        return super().__call__(x)


class MLPEncoder(Encoder, eqx.nn.MLP):
    """
    A MLP encoder.

    Parameters
    ----------
    edges_signs: `jax.Array`
        An array of +1s to denote tension and -1s to denote compression on the edges.
    q_shift: `float`, optional
        The minimum value of the latent representation.
    slice_out: `bool`, optional
        Whether to slice the output of the encoder to learn a mapping only
        w.r.t. a slice of the target shape.
    slice_indices: `jax.Array`, optional
        The indices of the points to slice from the target shape.
    in_size: `int`
        The dimension of the input.
    out_size: `int`
        The dimension of the output latents.
    width_size: `int`
        The size of the hidden layers.
    depth: `int`
        The number of hidden layers, including the output layer.
    activation: `Callable`
        The activation function for the hidden layers.
    final_activation: `Callable`
        The activation function for the output layer.
    key: `jax.random.PRNGKey`
        The random key.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        """
        Map a target shape to a latent representation.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.

        Returns
        -------
        q: `jax.Array`
            The latent representation.
        """
        # MLP prediction (must be positive due to softplus activation)
        q_hat = super().__call__(x)

        # NOTE: negative q denotes compression, positive tension.
        return (q_hat + self.q_shift) * self.edges_signs


# ===============================================================================
# Decoders
# ===============================================================================

class Decoder(eqx.Module):
    """
    A decoder.

    Parameters
    ----------
    load: `float`
        The area load applied to the structure.
    mask_edges: `jax.Array`
        A mask vector for the latent values to zero out.
    """
    load: Float
    mask_edges: Array

    def __init__(self, load, mask_edges, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load = load
        self.mask_edges = mask_edges

    def __call__(self, q, x, structure, aux_data=False):
        """
        Map a latent representation to a target shape.

        Parameters
        ----------
        q: `jax.Array`
            The latent representation.
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.
        aux_data: `bool`, optional
            Whether to return auxiliary data. The auxiliary data is a tuple of the
            force density parameters, the fixed node positions, and the applied loads.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        data: `tuple` of `jax.Array`
            The auxiliary data if `aux_data` is `True`.
        """
        # gather parameters
        q = self.get_q(q)
        xyz_fixed = self.get_xyz_fixed(x, structure)
        loads = self.get_loads(x, structure)

        # predict x
        x_hat = self.get_xyz((q, xyz_fixed, loads), structure)

        if aux_data:
            data = (q, xyz_fixed, loads)
            return x_hat, data

        return x_hat

    def get_q(self, q_hat):
        """
        Mask the latent values to zero out.

        Parameters
        ----------
        q_hat: `jax.Array`
            The latent representation.

        Returns
        -------
        q: `jax.Array`
            The masked latent representation.
        """
        return q_hat * self.mask_edges

    def get_xyz_fixed(self, x, structure):
        """
        Calculate the fixed vertex positions.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        xyz_fixed: `jax.Array`
            The fixed vertex positions.
        """
        indices = structure.indices_fixed
        x = jnp.reshape(x, (-1, 3))

        return x[indices, :]

    def get_loads(self, x, structure):
        """
        Calculate the applied vertex loads from a global area load.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        loads: `jax.Array`
            The applied vertex loads.
        """
        if self.load:
            return calculate_area_loads(x, structure, self.load)

        return jnp.zeros((structure.num_vertices, 3))

    def get_xyz(self, params, structure):
        """
        Lower level method to predict the target shape. It must be implemented by the subclasses.

        Parameters
        ----------
        params: `tuple` of `jax.Array`
            The parameters to predict the target shape from.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        raise NotImplementedError


# ===============================================================================
# Physics-based decoders
# ===============================================================================

class FDDecoder(Decoder):
    """
    A physics-based force density decoder.

    Parameters
    ----------
    model: `jax_fdm.EquilibriumModel`
        The force density model.
    load: `float`
        The area load applied to the structure.
    mask_edges: `jax.Array`
        A mask vector for the latent values to zero out.
    """
    model: EquilibriumModel

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        Predict the target shape from the simulation parameters.

        Parameters
        ----------
        params: `tuple` of `jax.Array`
            The parameters to predict the target shape from.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        q, xyz_fixed, loads = params
        # NOTE: to predict only free vertices, use instead
        # self.model.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       loads,
                                       structure)
        return jnp.ravel(x_hat)


class FDDecoderParametrized(FDDecoder):
    """
    A physics-based force density decoder that is directly optimizable.

    Parameters
    ----------
    q: `jax.Array`
        The initial force densities.
    model: `jax_fdm.EquilibriumModel`
        The force density model.
    load: `float`
        The area load applied to the structure.
    mask_edges: `jax.Array`
        A mask vector for the latent values to zero out.
    """
    q: Array

    def __init__(self, q, *args, **kwargs):
        self.q = q
        super().__init__(*args, **kwargs)

    def __call__(self, x, structure, aux_data=False, *args, **kwargs):
        """
        Solve equilibrium using stored force densities.

        Parameters
        ----------
        x: `jax.Array`
            The target shape (unused; force densities come from self.q).
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.
        aux_data: `bool`, optional
            Whether to return auxiliary data. The auxiliary data is a tuple of the
            force density parameters, the fixed node positions, and the applied loads.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        data: `tuple` of `jax.Array`
            The auxiliary data if `aux_data` is `True`.
        """
        return super().__call__(self.q, x, structure, aux_data)

    def predict_states(self, x, structure):
        """
        Predict equilibrium and parameter states for visualization.

        Parameters
        ----------
        x: `jax.Array`
            The target shape.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        eq_state: `jax_fdm.EquilibriumState`
            The current equilibrium state of the structure.
        fd_params_state: `jax_fdm.EquilibriumParametersState`
            The current state of simulation parameters.
        """
        # Predict shape
        x_hat, params = self(x, structure, True)

        return build_states(x_hat, params, structure)


# ===============================================================================
# Neural decoders
# ===============================================================================

class MLPDecoder(Decoder, eqx.nn.MLP):
    """
    A MLP decoder.

    Parameters
    ----------
    load: `float`
        The area load applied to the structure.
    mask_edges: `jax.Array`
        A mask vector for the latent values to zero out.
    in_size: `int`
        The dimension of the input.
    out_size: `int`
        The dimension of the output.
    width_size: `int`
        The size of the hidden layers.
    depth: `int`
        The number of hidden layers, including the output layer.
    activation: `Callable`
        The activation function for the hidden layers.
    key: `jax.random.PRNGKey`
        The random key.
    """
    # NOTE: Should the inheritance order be reversed?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        Map a latent representation to a target shape.

        Parameters
        ----------
        params: `tuple` of `jax.Array`
            The parameters to predict the target shape from. The parameters are
            the force density parameters, the fixed node positions, and the applied loads.
        structure: `jax_fdm.EquilibriumStructure`
            A structure with the discretization of the shape.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        # unpack parameters
        q, x_fixed, loads = params

        # predict x
        x_free = self._get_xyz(params)

        # Concatenate the position of the free and the fixed nodes
        indices = structure.indices_freefixed
        x_free = jnp.reshape(x_free, (-1, 3))
        x_hat = jnp.concatenate((x_free, x_fixed))[indices, :]

        return jnp.ravel(x_hat)

    def _get_xyz(self, params):
        """
        Map a latent representation to a target shape.

        Parameters
        ----------
        params: `tuple` of `jax.Array`
            The parameters to predict the target shape from. The parameters are
            the force density parameters, the fixed node positions, and the applied loads.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        # unpack parameters
        q, x_fixed, loads = params

        # NOTE: using this exotic way to call __call__ to map q to x due to multiple inheritance
        return eqx.nn.MLP.__call__(self, q)


class MLPDecoderXL(MLPDecoder):
    """
    A MLP decoder that maps latents and the boundary conditions (fixed positions and loads) to a shape.

    It assumes that the load has only a z-component, while x and y are always 0.

    Parameters
    ----------
    load: `float`
        The area load applied to the structure.
    mask_edges: `jax.Array`
        A mask vector for the latent values to zero out.
    in_size: `int`
        The dimension of the input.
    out_size: `int`
        The dimension of the output.
    width_size: `int`
        The size of the hidden layers.
    depth: `int`
        The number of hidden layers, including the output layer.
    activation: `Callable`
        The activation function for the hidden layers.
    key: `jax.random.PRNGKey`
        The random key.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_xyz(self, params):
        """
        Map a latent representation and the boundary conditions to a target shape.

        Parameters
        ----------
        params: `tuple` of `jax.Array`
            The parameters to predict the target shape from. The parameters are
            the force density parameters, the fixed node positions, and the applied loads.

        Returns
        -------
        x_hat: `jax.Array`
            The predicted shape.
        """
        # unpack parameters
        q, x_fixed, loads = params

        # concatenate long array
        x_fixed = jnp.ravel(x_fixed)
        loads_z = loads[:, 2]  # only z component, x and y are always 0
        params = jnp.concatenate((q, x_fixed, loads_z))

        return eqx.nn.MLP.__call__(self, params)


# ===============================================================================
# Helpers
# ===============================================================================

def build_states(x_hat, params, structure):
    """
    Assemble equilibrium and parameter states for visualization.

    Parameters
    ----------
    xyz_hat: `jax.Array`
        The predicted shape.
    params: `tuple` of `jax.Array`
        The parameters to predict the target shape from. The parameters are
        the force density parameters, the fixed node positions, and the applied loads.
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.

    Returns
    -------
    eq_state: `jax_fdm.EquilibriumState`
        The current equilibrium state of the structure.
    fd_params_state: `jax_fdm.EquilibriumParametersState`
        The current state of simulation parameters.
    """
    # Unpack aux data
    q, xyz_fixed, loads = params

    # Equilibrium parameters
    fd_params_state = calculate_fd_params_state(
        q,
        xyz_fixed,
        loads
    )

    # Equilibrium state
    x_hat = jnp.reshape(x_hat, (-1, 3))

    eq_state = calculate_equilibrium_state(
        q,
        x_hat,  # xyz_free | xyz_fixed
        loads,
        structure
    )

    return eq_state, fd_params_state
