"""
Optimize the force densities and shapes of a batch of target shapes starting from a pretrained model predictions (no vectorization).
"""
import os
from math import fabs
import yaml

import warnings

from time import perf_counter
from statistics import mean
from statistics import stdev

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

import equinox as eqx

from jaxopt import ScipyBoundedMinimize

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Polygon
from compas.geometry import Line

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fdm import DATA

from neural_fdm.builders import build_mesh_from_generator
from neural_fdm.builders import build_data_generator
from neural_fdm.builders import build_connectivity_structure_from_generator
from neural_fdm.builders import build_fd_decoder_parametrized
from neural_fdm.builders import build_loss_function
from neural_fdm.builders import build_neural_model

from neural_fdm.losses import print_loss_summary

from neural_fdm.serialization import load_model

from camera import CAMERA_CONFIG_BEZIER
from camera import CAMERA_CONFIG_TOWER

from optimize import calculate_params_init
from optimize import calculate_params_bounds


# ===============================================================================
# Script function
# ===============================================================================

def predict_optimize_batch(
        model_name,
        optimizer_name,
        task_name,
        blow=0.0,
        bup=20.0,
        maxiter=5000,
        tol=1e-6,
        seed=None,
        batch_size=None,
        verbose=True,
        save=False,
        view=False,
        slice=(0, -1),  # (50, 53) for bezier
        edgecolor=None,
):
    """
    Solve the prediction task on a batch target shapes with gradient-based optimization
    and box constraints, using a neural model to warmstart the optimization.
    The box constraints help generating compression-only or tension-only solutions.

    This script optimizes and visualizes. This is probably not the best idea, but oh well.

    Parameters
    ----------
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    optimizer_name: `str`
        The name gradient-based optimizer used to solve this task.
        Supported methods are slsqp and lbfgsb.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    blow: `float`, optional
        The lower bound of the box constraints on the model parameters.
        The bounds respect the force density signs of a task (compression or tension, currently hardcoded).        
    bup: `float`, optional
        The lower bound of the box constraints on the model parameters.
        The bounds respect the force density signs of a task (compression or tension, currently hardcoded).
    maxiter: `int`, optional
        The maximum number of optimization iterations.
    tol: `float`, optional
        The tolerance for the optimization.
    seed: `int` or `None`, optional
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`, optional
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    verbose: `bool`, optional
        If `True`, print to stdout intermediary results.
    save: `bool`, optional
        If `True`, save the predicted shapes as JSON files.
    view: `bool`, optional
        If `True`, view the predicted shapes.
    slice: `tuple`, optional
        The start and stop indices of the slice of the batch for saving and viewing.
        Defaults to all the shapes in the batch.        
    edgecolor: `str` or `None`, optional
        The color palette for the edges.
        Supported color palettes are fd to display force densities, and force to show forces.
        If `None`, the edges are colored by the force density in the shells tasks, and by the force in the tower tasks.
    """
    if edgecolor is None:
        if task_name == "bezier":
            EDGECOLOR = "fd"
        elif task_name == "tower":
            EDGECOLOR = "force"
    else:
        EDGECOLOR = edgecolor

    START, STOP = slice
    SAVE = save
    QMIN = blow
    QMAX = bup

    # pick camera configuration for task
    if task_name == "bezier":
        CAMERA_CONFIG = CAMERA_CONFIG_BEZIER
        _width = 900
    elif task_name == "tower":
        CAMERA_CONFIG = CAMERA_CONFIG_TOWER
        _width = 450

    # pick optimizer name
    optimizer_names = {"lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
    optimizer_name = optimizer_names[optimizer_name]

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
        batch_size = training_params["batch_size"]

    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']
    fd_params = config["fdm"]

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    compute_loss = build_loss_function(config, generator)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)

    # load model
    filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
    _model_name = model_name.split("_")[0]
    model_skeleton = build_neural_model(_model_name, config, generator, model_key)
    model = load_model(filepath, model_skeleton)

    # generate initial model parameters
    q0 = calculate_params_init(mesh, None, key, QMIN, QMAX)

    # create model
    print(f"Directly optimizing with {optimizer_name} using {model_name} init for {generator_name} dataset with {bounds_name} bounds on seed {seed}")
    decoder = build_fd_decoder_parametrized(q0, mesh, fd_params)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # split mode
    diff_decoder, static_decoder = eqx.partition(decoder, eqx.is_inexact_array)

    # wrap loss function to meet jax and jaxopt's ideosyncracies
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)  # Ensure this function is compiled at most once
    @eqx.filter_value_and_grad
    def compute_loss_diffable(diff_decoder, xyz_target):
        """
        """
        _decoder = eqx.combine(diff_decoder, static_decoder)
        return compute_loss(_decoder, structure, xyz_target, aux_data=False)

    # warmstart loss function to eliminate jit compilation time from perf measurements
    _ = compute_loss_diffable(diff_decoder, xyz_target=xyz_batch[None, 0])

    # define optimization function
    warnings.filterwarnings("ignore")

    opt = ScipyBoundedMinimize(
        fun=compute_loss_diffable,
        method=optimizer_name,
        jit=True,
        tol=tol,
        maxiter=maxiter,
        options={"disp": False},
        value_and_grad=True,
    )

    # define parameter bounds
    bound_low, bound_up = calculate_params_bounds(mesh, q0, QMIN, QMAX)
    bound_low_tree = eqx.tree_at(lambda tree: tree.q, diff_decoder, replace=(bound_low))
    bound_up_tree = eqx.tree_at(lambda tree: tree.q, diff_decoder, replace=(bound_up))
    bounds = (bound_low_tree, bound_up_tree)

    # optimize
    print("\nOptimizing shapes in sequence")
    opt_times = []
    loss_terms_batch = []

    were_successful = 0
    if STOP == -1:
        STOP = batch_size

    xyz_slice = xyz_batch[START:STOP]

    # Warmstart optimization
    _xyz_ = xyz_slice[0][None, :]
    start_time = perf_counter()
    diff_model_opt, opt_res = opt.run(diff_decoder, bounds, _xyz_)
    end_time = perf_counter() - start_time
    print(f"\tJIT compilation time (optimizer): {end_time:.4f} s")

    num_opts = xyz_slice.shape[0]
    for i, xyz in enumerate(xyz_slice):

        # get sample from batch, add extra dimension
        xyz = xyz[None, :]

        # predict with pretrained model
        if task_name == "bezier":
            q0 = model.encode(xyz.ravel())
        else:
            q0 = model.encode(xyz)

        # reinitialize decoders with pretrained model predictions
        decoder = eqx.tree_at(lambda tree: tree.q, decoder, replace=q0)
        diff_decoder = eqx.tree_at(lambda tree: tree.q, diff_decoder, replace=q0)

        # report start losses
        _, loss_terms = compute_loss(decoder, structure, xyz, aux_data=True)
        if verbose:
            print(f"Shape {i + 1}")
            print_loss_summary(loss_terms, prefix="\tStart")

        # optimize
        start_time = perf_counter()
        diff_model_opt, opt_res = opt.run(diff_decoder, bounds, xyz)
        opt_time = perf_counter() - start_time

        # unite optimal and static submodels
        model_opt = eqx.combine(diff_model_opt, static_decoder)

        # evaluate loss function at optimum point
        _, loss_terms = compute_loss(model_opt, structure, xyz, aux_data=True)
        if verbose:
            print_loss_summary(loss_terms, prefix="\tEnd")
            print(f"\tOpt success?: {opt_res.success}")
            print(f"\tOpt iters: {opt_res.iter_num}")
            print(f"\tOpt time: {opt_time:.4f} sec")

        if opt_res.success:
            were_successful += 1

        opt_times.append(opt_time)
        loss_terms_batch.append(loss_terms)

        # assemble datastructure for post-processing
        eqstate_hat, fd_params_hat = model_opt.predict_states(xyz, structure)
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)

        # export prediction
        if SAVE:
            filename = f"mesh_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # visualization
        if view:
            # create target mesh
            mesh_target = mesh.copy()
            _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
            for idx, key in mesh.index_key().items():
                mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

            viewer = Viewer(
                width=_width,
                height=900,
                show_grid=False,
                viewmode="lighted"
            )

            # modify view
            viewer.view.camera.position = CAMERA_CONFIG["position"]
            viewer.view.camera.target = CAMERA_CONFIG["target"]
            viewer.view.camera.distance = CAMERA_CONFIG["distance"]
            _rotation = CAMERA_CONFIG.get("rotation")
            if _rotation:
                viewer.view.camera.rotation = _rotation

            # edge colors
            if EDGECOLOR == "force":

                color_end = Color.from_rgb255(12, 119, 184)
                color_start = Color.white()
                cmap = ColorMap.from_two_colors(color_start, color_end)

                edgecolor = {}
                forces = [fabs(network_hat.edge_force(edge)) for edge in network_hat.edges()]
                fmin = min(forces)
                fmax = max(forces)

                for edge in network_hat.edges():
                    force = network_hat.edge_force(edge) * -1.0
                    if force < 0.0:
                        _color = Color.from_rgb255(227, 6, 75)
                    else:
                        value = (force - fmin) / (fmax - fmin)
                        _color = cmap(value)

                    edgecolor[edge] = _color
            else:
                edgecolor = EDGECOLOR

            viewer.add(
                network_hat,
                edgewidth=(0.01, 0.3),
                edgecolor=edgecolor,
                show_edges=True,
                edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                show_loads=False,
                loadscale=1.0,
                show_reactions=True,
                reactionscale=1.0,
                reactioncolor=Color.from_rgb255(0, 150, 10),
            )

            if task_name == "bezier":
                # target mesh
                viewer.add(
                    FDNetwork.from_mesh(mesh_target),
                    as_wireframe=True,
                    show_points=False,
                    linewidth=4.0,
                    color=Color.black().lightened()
                )

                # approximated mesh
                viewer.add(
                    mesh_hat,
                    show_points=False,
                    show_edges=False,
                    opacity=0.2
                )

            elif task_name == "tower":
                rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring in rings:
                    ring = Polygon(ring.tolist())
                    viewer.add(ring, opacity=0.5)

                lengths = []
                xyz_hat = model_opt(xyz, structure)
                rings_hat = jnp.reshape(xyz_hat, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring_a, ring_b in zip(rings, rings_hat):
                    for pt_a, pt_b in zip(ring_a, ring_b):
                        line = Line(pt_a, pt_b)
                        viewer.add(line)
                        lengths.append(line.length**2)

            # show le crème
            viewer.show()

    # report optimization statistics
    print(f"\nSuccessful optimizations: {were_successful}/{num_opts}")
    print(f"Optimization time over {num_opts} optimizations (s): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")

    labels = loss_terms_batch[0].keys()
    for label in labels:
        errors = [terms[label].item() for terms in loss_terms_batch]
        print(f"{label.capitalize()} over {num_opts} optimizations: {mean(errors):.4f} (+-{stdev(errors):.4f})")


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(predict_optimize_batch)
