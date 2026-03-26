from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax_fdm.equilibrium import EquilibriumMeshStructure, EquilibriumModel

from neural_fdm.generators import (
    BezierSurfaceAsymmetricPointGenerator,
    BezierSurfaceLerpPointGenerator,
    BezierSurfacePointGenerator,
    BezierSurfaceSymmetricDoublePointGenerator,
    BezierSurfaceSymmetricPointGenerator,
    CircularTubePointGenerator,
    EllipticalTubePointGenerator,
    TubePointGenerator,
)
from neural_fdm.losses import compute_loss, compute_loss_shell, compute_loss_shell_vae, compute_loss_tower
from neural_fdm.mesh import create_mesh_from_bezier_generator, create_mesh_from_tube_generator
from neural_fdm.models import (
    AutoEncoder,
    AutoEncoderPiggy,
    FDDecoder,
    FDDecoderParametrized,
    MLPDecoder,
    MLPDecoderXL,
    MLPEncoder,
)

# ===============================================================================
# Tower shape generator bounds
# ===============================================================================

def ellipse_minmax_values():
    """
    The boundary values (radius 1, radius 2, rotation) for a family of ellipses.

    Returns
    -------
    minval: `list` of `float`
        The minimum values for the ellipse.
    maxval: `list` of `float`
        The maximum values for the ellipse.

    Notes
    -----
    The radii are scale factors relative to the reference radius of a tower.
    """
    minval = [0.5, 0.5, 0.0]
    maxval = [1.5, 1.5, 0.0]

    return minval, maxval


def ellipse_rotated_minmax_values():
    """
    The boundary values (radius 1, radius 2, rotation) for a family of rotating ellipses.

    Returns
    -------
    minval: `list` of `float`
        The minimum values for the ellipse.
    maxval: `list` of `float`
        The maximum values for the ellipse.

    Notes
    -----
    The radii are scale factors relative to the reference radius of a tower.
    """
    minval = [0.5, 0.5, -15.0]
    maxval = [1.5, 1.5, 15.0]

    return minval, maxval


def get_tower_generator_minmax_values(name, bounds):
    """
    Get the minimum and maximum radii and rotation values for a tower generator.

    Parameters
    ----------
    name: `str`
        The name of the tower generator.
    bounds: `str`
        The name of the bounds to use.

    Returns
    -------
    minval: `jax.Array`
        The minimum values for the generator.
    maxval: `jax.Array`
        The maximum values for the generator.
    """
    experiments = {
        "straight": ellipse_minmax_values,
        "twisted": ellipse_rotated_minmax_values,
    }

    values_fn = experiments.get(bounds)
    if not values_fn:
        raise KeyError(f"Experiment bounds: {bounds} is currently unsupported!")

    # generate values on a quarter tile
    minval, maxval = values_fn()

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    return minval, maxval


# ===============================================================================
# Bezier shape generator bounds
# ===============================================================================

def pillow_minmax_values():
    """
    The boundary 3D coordinates for a family of pillow shapes.

    Returns
    -------
    minval: `list` of `list` of `float`
        The minimum values for the pillow on a quarter tile.
    maxval: `list` of `list` of `float`
        The maximum values for the pillow on a quarter tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def dome_minmax_values():
    """
    The boundary 3D coordinates for a family of dome shapes.

    Returns
    -------
    minval: `list` of `list` of `float`
        The minimum values for the dome on a quarter tile.
    maxval: `list` of `list` of `float`
        The maximum values for the dome on a quarter tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def saddle_minmax_values():
    """
    The boundary 3D coordinates for a family of saddle shapes.

    Returns
    -------
    minval: `list` of `list` of `float`
        The minimum values for the saddle on a quarter tile.
    maxval: `list` of `list` of `float`
        The maximum values for the saddle on a quarter tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 10.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def get_bezier_generator_minmax_values(name, bounds):
    """
    The boundary 3D coordinates for a family of Bezier shapes.

    Parameters
    ----------
    name: `str`
        The name of the Bezier generator.
    bounds: `str`
        The name of the bounds to use.

    Returns
    -------
    minval: `jax.Array`
        The minimum values for the generator.
    maxval: `jax.Array`
        The maximum values for the generator.
    """
    experiments = {
        "pillow": pillow_minmax_values,
        "dome": dome_minmax_values,
        "saddle": saddle_minmax_values
    }

    values_fn = experiments.get(bounds)
    if not values_fn:
        raise KeyError(f"Experiment bounds: {bounds} is currently unsupported!")

    # generate values on a quarter tile (assumes double symmetry)
    minval, maxval = values_fn()

    # concatenate bounds based on generator type and symmetry
    name_parts = name.split("_")

    # generator that blends between a symmetry and asymmetric surfaces
    if "lerp" in name_parts:
        return _get_bezier_generator_minmax_values_blend(minval, maxval)

    # generators with symmetry
    if "symmetric" in name_parts:
        if "double" not in name_parts:
            minval, maxval = _get_bezier_generator_minmax_values_symmetric(minval, maxval)
    elif "asymmetric" in name_parts:
        minval, maxval = _get_bezier_generator_minmax_values_asymmetric(minval, maxval)

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    return minval, maxval


def _get_bezier_generator_minmax_values_symmetric(minval, maxval):
    minval = minval + minval
    maxval = maxval + maxval

    return minval, maxval


def _get_bezier_generator_minmax_values_asymmetric(minval, maxval):
    minval = minval + minval + minval + minval
    maxval = maxval + maxval + maxval + maxval

    return minval, maxval


def _get_bezier_generator_minmax_values_blend(minval, maxval):
    minval_b, maxval_b = _get_bezier_generator_minmax_values_asymmetric(minval, maxval)

    minval_a = jnp.array(minval)
    maxval_a = jnp.array(maxval)

    minval_b = jnp.array(minval_b)
    maxval_b = jnp.array(maxval_b)

    return (minval_a, minval_b), (maxval_a, maxval_b)


# ===============================================================================
# Data generators
# ===============================================================================

def build_tube_point_generator(generator_params):
    """
    Build a generator that samples random points on a tube.

    Parameters
    ----------
    generator_params: `dict`
        The hyperparameters for the generator.

    Returns
    -------
    generator: `TubePointGenerator`
        The generator.
    """
    # unpack parameters
    name = generator_params["name"]
    bounds = generator_params["bounds"]

    height = generator_params["height"]
    radius = generator_params["radius"]
    num_sides = generator_params["num_sides"]
    num_levels = generator_params["num_levels"]
    num_rings = generator_params["num_rings"]

    # wiggle bounds for task
    minval, maxval = get_tower_generator_minmax_values(name, bounds)

    # Create data generator
    generators = {
        "ellipse": EllipticalTubePointGenerator,
        "circle": CircularTubePointGenerator,
    }

    name = generator_params["name"].split("_")[-1]
    generator = generators.get(name)
    if not generator:
        raise ValueError(f"Generator {name} is not supported yet!")

    return generator(height, radius, num_sides, num_levels, num_rings, minval, maxval)


def build_bezier_point_generator(generator_params):
    """
    Build a generator that samples random points on a Bezier surface.

    Parameters
    ----------
    generator_params: `dict`
        The hyperparameters for the generator.

    Returns
    -------
    generator: `BezierSurfacePointGenerator`
        The generator.
    """
    # unpack parameters
    name = generator_params["name"]
    num_u = generator_params["num_uv"]
    num_v = generator_params["num_uv"]
    size = generator_params["size"]
    num_pts = generator_params["num_points"]
    bounds_name = generator_params["bounds"]
    lerp_factor = generator_params.get("lerp_factor")

    # wiggle bounds for task
    minval, maxval = get_bezier_generator_minmax_values(name, bounds_name)

    # Create data generator
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # Create data generator
    generators = {
        "bezier_symmetric": BezierSurfaceSymmetricPointGenerator,
        "bezier_symmetric_double": BezierSurfaceSymmetricDoublePointGenerator,
        "bezier_asymmetric": BezierSurfaceAsymmetricPointGenerator,
        "bezier_lerp": BezierSurfaceLerpPointGenerator
    }

    generator = generators.get(name)
    if not generator:
        raise ValueError(f"Generator {name} is not supported yet!")

    return generator(size, num_pts, u, v, minval, maxval, lerp_factor)


def build_data_generator(config):
    """
    Build a generator that samples random points on a target shape.

    Parameters
    ----------
    config: `dict`
        The configuration for the generator.

    Returns
    -------
    generator: `PointGenerator`
        The generator.
    """
    # unpack parameters
    generator_params = config["generator"]

    # pick generator function
    generator_builders = {
        "bezier": build_bezier_point_generator,
        "tower": build_tube_point_generator
    }

    name = generator_params["name"].split("_")[0]

    generator_builder = generator_builders.get(name)
    if not generator_builder:
        raise ValueError(f"Generator {name} is not supported yet!")

    # build bezier generator
    return generator_builder(generator_params)


# ===============================================================================
# Mesh
# ===============================================================================

def build_mesh_from_generator(config, generator):
    """
    Generate a JAX FDM mesh according to the generator type.

    Parameters
    ----------
    config: `dict`
        The configuration for the generator.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    """
    if isinstance(generator, BezierSurfacePointGenerator):
        mesh_builder = create_mesh_from_bezier_generator
    elif isinstance(generator, TubePointGenerator):
        mesh_builder = create_mesh_from_tube_generator
    else:
        raise ValueError(f"Cannot make meshes with generator {generator}!")

    return mesh_builder(generator, config)


# ===============================================================================
# Structure (Graph)
# ===============================================================================

def build_connectivity_structure_from_generator(config, generator):
    """
    Build a structure from a generator.

    Parameters
    ----------
    config: `dict`
        The configuration for the generator.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    structure: `jax_fdm.EquilibriumStructure`
        A structure with the discretization of the shape.
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(config, generator)

    return EquilibriumMeshStructure.from_mesh(mesh)


# ===============================================================================
# Activation functions
# ===============================================================================

def get_activation_fn(name):
    """
    Fetch the activation function.

    Parameters
    ----------
    name: `str`
        The name of the activation function.

    Returns
    -------
    activation_fn: `Callable`
        The activation function.
    """
    functions = {
        "elu": jax.nn.elu,
        "relu": jax.nn.relu,
        "softplus": jax.nn.softplus
    }

    activation_fn = functions.get(name)
    if not activation_fn:
        raise KeyError(f"Activation name: {name} is currently unsupported!")

    return activation_fn


# ===============================================================================
# Optimizers
# ===============================================================================

def get_optimizer_fn(name):
    """
    Fetch the optimizer function.

    Parameters
    ----------
    name: `str`
        The name of the optimizer.

    Returns
    -------
    optimizer_fn: `Callable`
        The optimizer function.
    """
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
    }

    optimizer_fn = optimizers.get(name)
    if not optimizer_fn:
        raise KeyError(f"Optimize name: {name} is currently unsupported!")

    return optimizer_fn


def build_optimizer(config):
    """
    Construct an optimizer.

    Parameters
    ----------
    config: `dict`
        The configuration for the optimizer.

    Returns
    -------
    optimizer: `optax.GradientTransformation`
        The optimizer.
    """
    params = config["optimizer"]

    name = params["name"]
    learning_rate = params["learning_rate"]
    assert isinstance(learning_rate, float)

    optimizer_fn = get_optimizer_fn(name)
    optimizer = optimizer_fn(learning_rate=learning_rate)

    clip_norm = float(params["clip_norm"])
    if clip_norm:
        print(f"Optimizing with {name} with learning rate {learning_rate} and gradient clipping to global max norm of {clip_norm}")
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optimizer
        )
    else:
        print(f"Optimizing with {name} with learning rate {learning_rate}")

    return optimizer


# ===============================================================================
# Loss functions
# ===============================================================================

def build_loss_function(config, generator):
    """
    Build a loss function.

    Parameters
    ----------
    config: `dict`
        The configuration for the loss function.
    generator: `PointGenerator`
        A generator.

    Returns
    -------
    loss_fn: `Callable`
        The loss function.
    """
    task_name = config["generator"]["name"]
    loss_params = config["loss"]
    is_vae = "vae" in loss_params

    if "bezier" in task_name:
        _loss_fn = compute_loss_shell_vae if is_vae else compute_loss_shell

    elif "tower" in task_name:
        # Store the shape and height dimensions for the loss evaluation
        loss_params["shape"]["dims"] = generator.shape_tube
        loss_params["shape"]["levels_compression"] = generator.levels_rings_comp
        loss_params["shape"]["levels_tension"] = generator.levels_rings_tension

        _loss_fn = compute_loss_tower

    loss_fn = partial(
        compute_loss,
        loss_fn=_loss_fn,
        loss_params=loss_params
    )

    return loss_fn


# ===============================================================================
# Force density solver
# ===============================================================================

def build_fd_model():
    """
    Build a force density model.

    Returns
    -------
    fd_model: `jax_fdm.EquilibriumModel`
        The force density model.

    Notes
    -----
    This is a dense model, because batching rule is undefined to vectorize a sparse one.
    """
    fd_model = EquilibriumModel(
        tmax=1,
        eta=1e-6,
        is_load_local=False,
        itersolve_fn=None,
        implicit_diff=True,
        verbose=False
        )

    return fd_model


def calculate_edges_mask(mesh):
    """
    A mask to indicate what mesh edges are fully supported.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.

    Returns
    -------
    mask_edges: `jax.Array`
        The mask array with 1s for supported edges and 0s for unsupported edges.
    """
    mask_edges = []
    for edge in mesh.edges():
        mask_val = 1.0
        if mesh.is_edge_fully_supported(edge):
            mask_val = 0.0
            # NOTE: for tower task
            if mesh.edge_attribute(edge, "tag") == "ring":
                if not mesh.is_edge_on_boundary(*edge):
                    mask_val = 1.0
        mask_edges.append(mask_val)

    return jnp.array(mask_edges, dtype=jnp.int64)


def calculate_edges_stress_signs(mesh):
    """
    Calculate an integer array to indicate what mesh edges are in compression and in tension.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.

    Returns
    -------
    signs: `jax.Array`
        The array with -1s for compression and 1s for tension.
    """
    signs = []
    for edge in mesh.edges():
        sign = -1  # compression by default
        # NOTE: for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1
        signs.append(sign)

    return jnp.array(signs, dtype=jnp.int64)


# ===============================================================================
# Decoders
# ===============================================================================

def build_fd_decoder_parametrized(q0, mesh, params):
    """
    Build a force density decoder for direct optimization.

    Parameters
    ----------
    q0: `jax.Array`
        The initial force densities.
    mesh: `jax_fdm.FDMesh`
        The mesh.
    params: `dict`
        The hyperparameters for the decoder.

    Returns
    -------
    decoder: `eqx.Module`
        The force density decoder.
    """
    # unpack hyperparams
    load = params["load"]

    # create FD model
    fd_model = build_fd_model()

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    decoder = FDDecoderParametrized(
        q0,
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_fd_decoder(mesh, params):
    """
    Build a force density decoder to connect with a neural encoder.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    params: `dict`
        The hyperparameters for the decoder.

    Returns
    -------
    decoder: `eqx.Module`
        The force density decoder.
    """
    # unpack hyperparams
    load = params["load"]

    # create FD model
    fd_model = build_fd_model()

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    decoder = FDDecoder(
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_neural_decoder(mesh, key, params):
    """
    Build an MLP decoder.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the decoder.

    Returns
    -------
    decoder: `eqx.Module`
        The decoder.
    """
    # unpack hyperparameters
    nn_params, fd_params = params

    # get neural network params
    include_xl = nn_params["include_params_xl"]
    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]

    # get load
    load = fd_params["load"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()
    num_vertices_free = len(list(mesh.vertices_free()))
    num_vertices_fixed = len(list(mesh.vertices_fixed()))

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # define size of input layer
    in_size = num_edges
    decoder_cls = MLPDecoder

    if include_xl:
        in_size += num_vertices_fixed * 3
        in_size += num_vertices
        decoder_cls = MLPDecoderXL

    # instantiate MLP
    decoder = decoder_cls(
        load=load,
        mask_edges=mask_edges,
        in_size=in_size,
        out_size=num_vertices_free * 3,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        key=key
        )

    return decoder


# ===============================================================================
# Encoders
# ===============================================================================

def build_neural_encoder(mesh, key, params, generator):
    """
    Build an encoder (MLP or GNN based on config).

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the encoder.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    encoder: `eqx.Module`
        The encoder.
    """
    # unpack hyperparameters
    nn_params, generator_params = params

    # Dispatch to GNN encoder if configured
    encoder_type = nn_params.get("encoder_type", "mlp")
    if encoder_type == "gnn":
        return build_gnn_encoder(mesh, key, params, generator)

    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]
    final_activation_name = nn_params["final_activation_fn_name"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()

    # get edges stress signs
    edges_signs = calculate_edges_stress_signs(mesh)

    # define input size
    in_size = num_vertices
    is_tower_task = "tower" in generator_params["name"]
    if is_tower_task:
        in_size = generator_params["num_rings"] * generator_params["num_sides"]
    in_size *= 3

    # define slices
    slice_out = False
    slice_indices = None
    if is_tower_task:
        slice_out = True
        slice_indices = generator.indices_rings_comp_ravel

    # q shift
    q_shift = nn_params["shift"]

    # instantiate MLP
    encoder = MLPEncoder(
        edges_signs=edges_signs,
        q_shift=q_shift,
        slice_out=slice_out,
        slice_indices=slice_indices,
        in_size=in_size,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        final_activation=get_activation_fn(final_activation_name),
        key=key
        )

    return encoder


def build_gnn_encoder(mesh, key, params, generator):
    """
    Build a GNN encoder for variable-topology form-finding.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the encoder.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    encoder: `GNNEncoder`
        The GNN encoder.
    """
    from neural_fdm.gnn import GNNEncoder
    from neural_fdm.graph import edge_index_from_mesh

    nn_params, generator_params = params

    hidden_layer_size = nn_params.get("hidden_layer_size", 128)
    num_layers = nn_params.get("hidden_layer_num", 4)
    q_shift = nn_params.get("shift", 0.0)

    mesh.number_of_edges()
    edges_signs = calculate_edges_stress_signs(mesh)
    edge_index = edge_index_from_mesh(mesh)

    is_tower_task = "tower" in generator_params["name"]
    slice_out = False
    slice_indices = None
    if is_tower_task:
        slice_out = True
        slice_indices = generator.indices_rings_comp_ravel

    encoder = GNNEncoder(
        edges_signs=edges_signs,
        q_shift=q_shift,
        slice_out=slice_out,
        slice_indices=slice_indices,
        node_feat_dim=3,
        edge_feat_dim=4,
        hidden_dim=hidden_layer_size,
        num_layers=num_layers,
        edge_index=edge_index,
        key=key,
    )

    return encoder


def build_variational_encoder(mesh, key, params, generator):
    """Build a variational MLP encoder for VAE form-finding.

    Follows the same parameter extraction as build_neural_encoder
    but returns a VariationalMLPEncoder with mu/log_sigma heads.
    """
    from neural_fdm.variational import VariationalMLPEncoder

    nn_params, generator_params = params
    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]

    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()
    edges_signs = calculate_edges_stress_signs(mesh)

    in_size = num_vertices
    is_tower_task = "tower" in generator_params["name"]
    if is_tower_task:
        in_size = generator_params["num_rings"] * generator_params["num_sides"]
    in_size *= 3

    slice_out = False
    slice_indices = None
    if is_tower_task:
        slice_out = True
        slice_indices = generator.indices_rings_comp_ravel

    q_shift = nn_params["shift"]

    return VariationalMLPEncoder(
        edges_signs=edges_signs,
        q_shift=q_shift,
        slice_out=slice_out,
        slice_indices=slice_indices,
        in_size=in_size,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        key=key,
    )


def build_variational_formfinder(mesh, key, params, generator):
    """Build a VAE formfinder: variational encoder + FDM decoder.

    The FDM decoder is identical to the deterministic formfinder.
    Only the encoder changes to produce distribution parameters.

    Reference: Pastrana et al. (ICLR 2025), Section 6.1.
    """
    from neural_fdm.variational import VariationalAutoEncoder

    nn_params, fd_params = params
    k1, k2 = jax.random.split(key)
    encoder = build_variational_encoder(mesh, k1, nn_params, generator)
    decoder = build_fd_decoder(mesh, fd_params)
    return VariationalAutoEncoder(encoder, decoder)


# ===============================================================================
# Autoencoder models
# ===============================================================================

def build_neural_formfinder(mesh, key, params, generator):
    """
    Instantiate an autoencoder model with a neural encoder and a mechanical decoder.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the model.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    model: `eqx.Module`
        The autoencoder model.
    """
    # Unpack hyperparams
    nn_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, nn_params, generator)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder(mesh, key, params, generator):
    """
    Instantiate a fully neural autoencoder model.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the model.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    model: `eqx.Module`
        The autoencoder model.
    """
    # Unpack hyperparams
    enc_params, dec_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params, generator)

    # Build MLP decoder
    decoder = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder_piggy(mesh, key, params, generator):
    """
    Instantiate an autoencoder model with a piggybacking neural decoder.

    Parameters
    ----------
    mesh: `jax_fdm.FDMesh`
        The mesh.
    key: `jax.random.PRNGKey`
        The random key.
    params: `dict`
        The hyperparameters for the model.
    generator: `PointGenerator`
        The generator.

    Returns
    -------
    model: `eqx.Module`
        The autoencoder model.
    """
    # Unpack hyperparams

    enc_params, dec_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params, generator)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Build MLP decoder
    decoder_piggy = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoderPiggy(encoder, decoder, decoder_piggy)

    return model


def build_neural_model(name, config, generator, model_key):
    """
    Build a neural model.

    Parameters
    ----------
    name: `str`
        The name of the model.
    config: `dict`
        The configuration for the model.
    generator: `PointGenerator`
        The generator.
    model_key: `jax.random.PRNGKey`
        The random key.

    Returns
    -------
    model: `eqx.Module`
        The autoencoder model.
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(config, generator)

    # build model
    fd_params = config["fdm"]
    decoder_params = config["decoder"]
    encoder_params = config["encoder"]
    generator_params = config["generator"]

    encoder_params = (encoder_params, generator_params)

    # select model
    if name == "formfinder":
        build_fn = build_neural_formfinder
        params = (encoder_params, fd_params)
    elif name == "autoencoder":
        build_fn = build_neural_autoencoder
        params = (encoder_params, (decoder_params, fd_params))
    elif name == "piggy":
        build_fn = build_neural_autoencoder_piggy
        params = (encoder_params, (decoder_params, fd_params), fd_params)
    elif name == "variational_formfinder":
        build_fn = build_variational_formfinder
        params = (encoder_params, fd_params)
    else:
        raise ValueError(f"Model name {name} is unsupported")

    return build_fn(mesh, model_key, params, generator)
