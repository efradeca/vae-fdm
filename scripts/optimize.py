"""
Optimize force densities to match a batch of target shapes with gradient-based optimization, one shape at a time (no vectorization).
"""
import os
from math import fabs
import yaml

import warnings

import matplotlib.pyplot as plt

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
from compas.geometry import Polyline
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fdm import DATA

from neural_fdm.builders import build_mesh_from_generator
from neural_fdm.builders import build_data_generator
from neural_fdm.builders import build_connectivity_structure_from_generator
from neural_fdm.builders import build_fd_decoder_parametrized
from neural_fdm.builders import build_loss_function

from neural_fdm.losses import print_loss_summary

from camera import CAMERA_CONFIG_BEZIER
from camera import CAMERA_CONFIG_TOWER

from shapes import BEZIERS


# ===============================================================================
# Script function
# ===============================================================================

def optimize_batch(
        optimizer,
        task_name,
        shape_name=None,
        param_init=None,
        blow=0.0,
        bup=20.0,
        maxiter=5000,
        tol=1e-6,
        seed=None,
        batch_size=None,
        slice=(0, -1),
        save=False,
        view=False,
        edgecolor=None,
        show_reactions=False,
        edgewidth=(0.01, 0.25),
        fmax=None,
        fmax_tens=None,
        fmax_comp=None,
        qmin=None,
        qmax=None,
        verbose=True,
        record=False,
        save_metrics=False,
):
    """
    Solve the prediction task on a batch target shapes with gradient-based optimization and box constraints.
    The box constraints help generating compression-only or tension-only solutions.

    This script optimizes and visualizes. This is probably not the best idea, but oh well.

    Parameters
    ----------
    optimizer: `str`
        The name gradient-based optimizer used to solve this task.
        Supported methods are slsqp and lbfgsb.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    shape_name: `str` or `None`, optional
        The name of the shape to optimize.
        Supported shell shapes are pillow, dome, saddle, hypar, pringle, and cannon;
        and require of a `bezier_symmetric_double` generator.
        Supported tower shapes are either named by an integer or a float scalar.
        If the name is an integer, the generator should be `tower_ellipse`, and `tower_circle` if the name is a float.
        In general, if a name is provided, the optimization is performed on this shape, ignoring the batch.
    param_init: `float` or `None`, optional
        If specified, it determines the starting value of all the model parameters.
        If `None`, then it samples parameters between `blow` and `bup` from a uniform distribution.
        The sampling respects the force density signs of a task (compression or tension, currently hardcoded).
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
        If `None`, it defaults to the task hyperparameters file.
    batch_size: `int` or `None`, optional
        The size of the batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    slice: `tuple`, optional
        The start and stop indices of the slice of the batch for saving and viewing.
        Defaults to all the shapes in the batch.
    save: `bool`, optional
        If `True`, save the predicted shapes as JSON files.
    view: `bool`, optional
        If `True`, view the predicted shapes.
    show_reactions: `bool`, optional
        If `True`, show the reactions on the predicted shapes upon display.
    edgewidth: `tuple`, optional
        The minimum and maximum width of the edges for visualization.
    fmax: `float` or `None`, optional
        The maximum force for the visualization.
    fmax_tens: `float` or `None`, optional
        The maximum tensile force for the visualization.
    fmax_comp: `float` or `None`, optional
        The maximum compressive force for the visualization.
    qmin: `float` or `None`, optional
        The minimum force density for the visualization.
    qmax: `float` or `None`, optional
        The maximum force density for the visualization.
    verbose: `bool`, optional
        If `True`, print to stdout intermediary results.
    record: `bool`, optional
        If `True`, record the loss history.
    edgecolor: `str` or `None`, optional
        The color palette for the edges.
        Supported color palettes are fd to display force densities, and force to show forces.
        If `None`, the edges are colored by the force density in the shells tasks, and by the force in the tower tasks.
    save_metrics: `bool`, optional
        If `True`, saves the calcualted batch metrics in text files.
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
    EDGEWIDTH = edgewidth

    # pick camera configuration for task
    if task_name == "bezier":
        CAMERA_CONFIG = CAMERA_CONFIG_BEZIER
        _width = 900
    elif task_name == "tower":
        CAMERA_CONFIG = CAMERA_CONFIG_TOWER
        _width = 450

    # pick optimizer name
    optimizer_names = {"lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
    optimizer_name = optimizer_names[optimizer]

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
    _, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    compute_loss = build_loss_function(config, generator)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)

    # generate initial model parameters
    q0 = calculate_params_init(mesh, param_init, key, QMIN, QMAX)

    # create model
    print(f"Directly optimizing with {optimizer_name} for {generator_name} dataset with {bounds_name} bounds on seed {seed}")
    model = build_fd_decoder_parametrized(q0, mesh, fd_params)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # split model
    diff_model, static_model = eqx.partition(model, eqx.is_inexact_array)

    # wrap loss function to meet jax and jaxopt's ideosyncracies
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)  # ensure this function is compiled at most once
    @eqx.filter_value_and_grad
    def compute_loss_diffable(diff_model, xyz_target):
        """
        """
        _model = eqx.combine(diff_model, static_model)
        return compute_loss(_model, structure, xyz_target, aux_data=False)

    # warmstart loss function to eliminate jit compilation time from perf measurements
    start_time = perf_counter()
    _ = compute_loss_diffable(diff_model, xyz_target=xyz_batch[None, 0])
    end_time = perf_counter() - start_time
    print(f"JIT compilation time (loss): {end_time:.4f} s")

    # define callback function
    history = []
    recorder = lambda x: history.append(x) if record else None

    opt = ScipyBoundedMinimize(
        fun=compute_loss_diffable,
        method=optimizer_name,
        jit=True,
        tol=tol,
        maxiter=maxiter,
        options={"disp": False},
        value_and_grad=True,
        callback=recorder
    )

    # disable scipy warnings about hitting the box constraints
    warnings.filterwarnings("ignore")

    # define parameter bounds
    bound_low, bound_up = calculate_params_bounds(mesh, q0, QMIN, QMAX)
    bound_low_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_low))
    bound_up_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_up))
    bounds = (bound_low_tree, bound_up_tree)

    # optimize
    print("\nOptimizing shapes in sequence")
    qs = []
    opt_times = []
    loss_terms_batch = []

    were_successful = 0
    if STOP == -1:
        STOP = batch_size

    xyz_slice = xyz_batch[START:STOP]

    # sample target points from prescribed shape name
    if shape_name is not None and "bezier" in task_name:
        transform = BEZIERS[shape_name]
        transform = jnp.array(transform)
        xyz = generator.evaluate_points(transform)
        xyz_slice = xyz[None, :]

    # Warmstart optimization
    _xyz_ = xyz_batch[0][None, :]
    start_time = perf_counter()
    diff_model_opt, opt_res = opt.run(diff_model, bounds, _xyz_)
    end_time = perf_counter() - start_time
    print(f"\tJIT compilation time (optimizer): {end_time:.4f} s")

    num_opts = 0
    for i, xyz in enumerate(xyz_slice):

        num_opts += 1
        xyz = xyz[None, :]

        # report start losses
        _, loss_terms = compute_loss(model, structure, xyz, aux_data=True)
        if verbose:
            print(f"\nShape {i + 1}")
            print_loss_summary(loss_terms, prefix="\tStart")

        # optimize
        start_time = perf_counter()
        diff_model_opt, opt_res = opt.run(diff_model, bounds, xyz)
        opt_time = perf_counter() - start_time

        # unite optimal and static submodels
        model_opt = eqx.combine(diff_model_opt, static_model)

        # assemble datastructure for post-processing
        eqstate_hat, fd_params_hat = model_opt.predict_states(xyz, structure)
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)

        # evaluate loss function at optimum point
        _, loss_terms = compute_loss(model_opt, structure, xyz, aux_data=True)

        if verbose:
            print_loss_summary(loss_terms, prefix="\tEnd")
            print(f"\tOpt success?: {opt_res.success}")
            print(f"\tOpt iters: {opt_res.iter_num}")
            print(f"\tOpt time: {opt_time:.4f} sec")

        if record:
            _losses = []
            for xk in history:
                _loss, _ = compute_loss_diffable(xk, xyz)
                _losses.append(_loss)

            plt.figure()
            plt.plot(jnp.array(_losses))
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid()
            plt.show()

        if opt_res.success:
            were_successful += 1

        qs.extend([_q.item() for _q in fd_params_hat.q])
        opt_times.append(opt_time)
        loss_terms_batch.append(loss_terms)

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

            # edges to view
            # NOTE: we are not visualizing edges on boundaries since they are supported
            edges_2_view = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]

            # compute stats
            forces_all = []
            forces_comp_all = []
            forces_tens_all = []

            for edge in network_hat.edges():

                force = network_hat.edge_force(edge)
                force_abs = fabs(force)
                forces_all.append(force_abs)
                if force <= 0.0:
                    forces_comp_all.append(force_abs)
                else:
                    forces_tens_all.append(force_abs)

            fmin = 0.0
            fmin_comp = 0.0
            fmin_tens = 0.0
            if fmax is None:
                fmax = max(forces_all)
            if fmax_tens is None:
                if forces_tens_all:
                    fmax_tens = max(forces_tens_all)
                else:
                    fmax_tens = 0.0
            if fmax_comp is None:
                if forces_comp_all:
                    fmax_comp = max(forces_comp_all)
                else:
                    fmax_comp = 0.0

            # edge width
            width_min, width_max = EDGEWIDTH
            _forces = [fabs(mesh_hat.edge_force(edge)) for edge in mesh_hat.edges()]
            _forces = remap_values(_forces, original_min=fmin, original_max=fmax)
            _widths = remap_values(_forces, width_min, width_max)
            edgewidth = {edge: width for edge, width in zip(mesh_hat.edges(), _widths)}

            # edge colors
            edgecolor = EDGECOLOR
            if edgecolor == "force":
                edgecolor = {}

                color_start = Color.white()
                color_comp_end = Color.from_rgb255(12, 119, 184)
                cmap_comp = ColorMap.from_two_colors(color_start, color_comp_end)
                color_tens_end = Color.from_rgb255(227, 6, 75)
                cmap_tens = ColorMap.from_two_colors(color_start, color_tens_end)

                for edge in mesh_hat.edges():

                    force = mesh_hat.edge_force(edge)

                    if force == 0.0:
                        edgecolor[edge] = color_start
                    else:
                        if force < 0.0:
                            _cmap = cmap_comp
                            _fmin = fmin_comp
                            _fmax = fmax_comp
                        else:
                            _cmap = cmap_tens
                            _fmin = fmin_tens
                            _fmax = fmax_tens

                        value = (fabs(force) - _fmin) / (_fmax - _fmin)
                        edgecolor[edge] = _cmap(value)

            viewer.add(
                network_hat,
                edgewidth=edgewidth,
                edgecolor=edgecolor,
                show_edges=True,
                edges=edges_2_view,
                nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                show_loads=False,
                loadscale=1.0,
                show_reactions=show_reactions,
                reactionscale=1.0,
                reactioncolor=Color.from_rgb255(0, 150, 10),
            )

            viewer.add(
                mesh_hat,
                show_points=False,
                show_edges=False,
                opacity=0.7,
                color=Color.grey().lightened(100),
            )

            for _vertices in mesh.vertices_on_boundaries():
                viewer.add(
                    Polyline([mesh_hat.vertex_coordinates(vkey) for vkey in _vertices]),
                    linewidth=4.0,
                    color=Color.black().lightened()
                    )

            # show le crème
            viewer.show()

    # report statistics
    print(f"\nSuccessful optimizations: {were_successful}/{num_opts}")
    if num_opts > 1:
        print(f"Optimization time over {num_opts} optimizations (s): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")

        labels = loss_terms_batch[0].keys()
        for label in labels:
            errors = [terms[label].item() for terms in loss_terms_batch]
            print(f"{label.capitalize()} over {num_opts} optimizations: {mean(errors):.4f} (+-{stdev(errors):.4f})")

    if save_metrics:
        # Export force densities
        filename = f"{optimizer}_{task_name}_q_eval.txt"
        filepath = os.path.join(DATA, filename)

        metrics = [f"{_q}\n" for _q in qs]
        with open(filepath, 'w') as output:
            output.writelines(metrics)

        print(f"Saved batch qs to {filepath}")


# ===============================================================================
# Helper functions
# ===============================================================================

def calculate_params_init(mesh, param_init, key, minval, maxval):
    """
    Calculate the initial force densities for the optimization.

    Parameters
    ----------
    mesh: `compas.datastructures.Mesh`
        The mesh to optimize.
    param_init: `float` or `None`
        If specified, it determines the starting value of all the model parameters.
        If `None`, then it samples parameters between `b_low` and `b_up` from a uniform distribution.        
    key: `jax.random.PRNGKey`
        The random seed for the uniform distribution.
    minval: `float`
        The minimum value for the uniform distribution.
    maxval: `float`
        The maximum value for the uniform distribution.

    Returns
    -------
    q0: `jax.numpy.ndarray`
        The initial force densities.
    """
    num_edges = mesh.number_of_edges()

    signs = []
    for edge in mesh.edges():
        sign = -1.0  # compression by default
        # for tower task
        # tag-based sign assignment for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1.0
        signs.append(sign)

    signs = jnp.array(signs)

    if param_init is not None:
        q0 = jnp.ones(num_edges) * param_init
    else:
        q0 = jrn.uniform(key, shape=(num_edges, ), minval=minval, maxval=maxval)

    return q0 * signs


def calculate_params_bounds(mesh, q0, minval, maxval):
    """
    Calculate the box constraints for the optimization.

    Parameters
    ----------
    mesh: `compas.datastructures.Mesh`
        The mesh to optimize.
    q0: `jax.numpy.ndarray`
        The initial force densities.
    minval: `float`
        The value of the lower bound.
    maxval: `float`
        The value of the upper bound.

    Returns
    -------
    bound_low: `jax.numpy.ndarray`
        The lower box constraint.
    bound_up: `jax.numpy.ndarray`
        The upper box constraint.
    """
    bound_low = []
    bound_up = []
    for edge in mesh.edges():
        # compression by default
        b_low = maxval * -1.0
        b_up = minval * -1.0
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            b_low = minval
            b_up = maxval

        bound_low.append(b_low)
        bound_up.append(b_up)

    bound_low = jnp.array(bound_low)
    bound_up = jnp.array(bound_up)

    return bound_low, bound_up


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(optimize_batch)
