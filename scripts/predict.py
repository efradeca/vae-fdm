"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""
import os
from functools import partial
from math import fabs
import yaml

from time import perf_counter
from statistics import mean
from statistics import stdev

import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

import equinox as eqx

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Polygon
from compas.geometry import Line
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fdm import DATA

from neural_fdm.builders import build_loss_function
from neural_fdm.builders import build_mesh_from_generator
from neural_fdm.builders import build_data_generator
from neural_fdm.builders import build_connectivity_structure_from_generator
from neural_fdm.builders import build_neural_model

from neural_fdm.losses import print_loss_summary

from neural_fdm.serialization import load_model

from camera import CAMERA_CONFIG_BEZIER
from camera import CAMERA_CONFIG_TOWER

from train import count_model_params


# ===============================================================================
# Script function
# ===============================================================================

def predict_batch(
        model_name,
        task_name,
        seed=None,
        batch_size=None,
        time_batch_inference=False,
        predict_in_sequence=True,
        slice=(0, -1),  # (50, 53) for bezier
        view=False,
        save=False,
        save_metrics=False,
        edgecolor=None,
):
    """
    Predict a batch of target shapes with a pretrained model.

    Parameters
    ----------
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    seed: `int` or `None`, optional
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    batch_size: `int` or `None`, optional
        The size of the batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    time_batch_inference: `bool`, optional
        If `True`, report the inference time over a data batch.
    predict_in_sequence: `bool`, optional
        If `True`, predict every shape in the prescribed slice of the data batch, one at a time.
    slice: `tuple`, optional
        The start and stop indices of the slice of the batch for saving and viewing.
        Defaults to all the shapes in the batch.        
    view: `bool`, optional
        If `True`, view the predicted shapes.
    save: `bool`, optional
        If `True`, saves the predicted shapes as JSON files.
    save_metrics: `bool`, optional
        If `True`, saves the calculated batch metrics in text files.
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

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
        batch_size = training_params["batch_size"]

    if STOP == -1:
        STOP = batch_size

    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)
    compute_loss = build_loss_function(config, generator)

    # print info
    print(f"Making predictions with {model_name} on {generator_name} dataset with {bounds_name} bounds\n")
    print(f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges")

    # load model
    filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
    _model_name = model_name.split("_")[0]
    model_skeleton = build_neural_model(_model_name, config, generator, model_key)
    model = load_model(filepath, model_skeleton)
    print(f"Model parameter count: {count_model_params(model)}")

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # inference function to time
    timed_fn = jit(vmap(partial(model, structure=structure)))

    # NOTE: Using eqx.debug to ensure this function is compiled at most once
    timed_fn = vmap(partial(model, structure=structure))
    timed_fn = eqx.debug.assert_max_traces(timed_fn, max_traces=1)
    timed_fn = jit(timed_fn)

    # time inference time on full batch
    if time_batch_inference:

        # warmstart
        timed_fn(xyz_batch)

        # time
        times = []
        for i in range(10):
            start = perf_counter()
            timed_fn(xyz_batch).block_until_ready()
            duration = 1000.0 * (perf_counter() - start)  # time in milliseconds
            times.append(duration)
        print(f"Inference time on batch size {batch_size}: {mean(times):.5f} (+-{stdev(times):.5f}) ms")

    # report batch losses
    _, loss_terms = compute_loss(model, structure, xyz_batch, aux_data=True)
    print_loss_summary(loss_terms, prefix="Batch\t")

    # make individual predictions
    if not predict_in_sequence:
        return

    print("\nPredicting shapes in sequence")
    qs = []
    opt_times = []
    loss_terms_batch = []
    num_predictions = 0

    # warmstart again, just in case
    _xyz_ = xyz_batch[0][None, :]
    start_time = perf_counter()
    timed_fn(_xyz_).block_until_ready()
    end_time = perf_counter() - start_time
    print(f"JIT compilation time: {end_time * 1000.0:.2f} ms")

    start = perf_counter()
    for i in range(START, STOP):

        xyz = xyz_batch[i]
        _xyz = xyz[None, :]

        # do inference on one design
        start_time = perf_counter()
        timed_fn(_xyz).block_until_ready()
        end_time = perf_counter() - start_time  # time in seconds
        opt_times.append(end_time)
        num_predictions += 1

        # calculate loss
        _, loss_terms = compute_loss(
            model,
            structure,
            xyz[None, :],
            aux_data=True
        )

        # predict equilibrium states for viz and i/o
        eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)

        # print loss statistics
        loss_terms_batch.append(loss_terms)
        loss_terms["time"] = jnp.array([end_time])
        print_loss_summary(loss_terms, prefix=f"Shape {i} \t")

        # extract additional statistics
        qs.extend([_q.item() for _q in fd_params_hat.q])

        if view or save:
            # assemble datastructure for post-processing
            network_hat = FDNetwork.from_mesh(mesh_hat)
            network_hat.print_stats()
            print()

        # export prediction
        if save:
            filename = f"mesh_{model_name}_{task_name}_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # Create target mesh
        mesh_target = mesh.copy()
        _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
        for idx, key in mesh.index_key().items():
            mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

        # visualization
        if view:
            # pick camera configuration for task
            if task_name == "bezier":
                _width = 900
                CAMERA_CONFIG = CAMERA_CONFIG_BEZIER
            elif task_name == "tower":
                _width = 450
                CAMERA_CONFIG = CAMERA_CONFIG_TOWER

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

            elif task_name == "tower" and EDGECOLOR == "fd":
                edgecolor = {}
                cmap = ColorMap.from_mpl("viridis")
                _edges = [edge for edge in network_hat.edges() if mesh.edge_attribute(edge, "tag") == "cable"]
                values = [fabs(mesh_hat.edge_forcedensity(edge)) for edge in _edges]
                ratios = remap_values(values)
                edgecolor = {edge: cmap(ratio) for edge, ratio in zip(_edges, ratios)}
                for edge in network_hat.edges():
                    if mesh.edge_attribute(edge, "tag") != "cable":
                        edgecolor[edge] = Color.pink()

            else:
                edgecolor = EDGECOLOR

            vertices_2_view = list(mesh.vertices())
            color_load = Color.from_rgb255(0, 150, 10)
            reactioncolor = color_load
            show_reactions = True
            if task_name == "bezier":
                if EDGECOLOR == "fd":
                    show_reactions = False
                vertices_2_view = []
                for vkey in mesh.vertices():
                    if len(mesh.vertex_neighbors(vkey)) < 3:
                        continue
                    vertices_2_view.append(vkey)

                reactioncolor = {}
                for vkey in mesh.vertices():
                    _color = Color.pink()
                    if mesh.is_vertex_on_boundary(vkey):
                        _color = color_load
                    reactioncolor[vkey] = _color

            viewer.add(
                network_hat,
                edgewidth=(0.01, 0.25),
                edgecolor=edgecolor,
                show_edges=True,
                edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                nodes=vertices_2_view,
                show_loads=False,
                loadscale=1.0,
                show_reactions=show_reactions,
                reactionscale=1.0,
                reactioncolor=reactioncolor
            )

            if task_name == "bezier":
                # approximated mesh
                viewer.add(
                     mesh_hat,
                     show_points=False,
                     show_edges=False,
                     opacity=0.1
                 )

                # target mesh
                viewer.add(
                    FDNetwork.from_mesh(mesh_target),
                    as_wireframe=True,
                    show_points=False,
                    linewidth=4.0,
                    color=Color.black().lightened()
                    )

            elif task_name == "tower":
                rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring in rings:
                    ring = Polygon(ring.tolist())
                    viewer.add(ring, opacity=0.5)

                xyz_hat = model(xyz, structure)
                rings_hat = jnp.reshape(xyz_hat, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring_a, ring_b in zip(rings, rings_hat):
                    for pt_a, pt_b in zip(ring_a, ring_b):
                        viewer.add(Line(pt_a, pt_b))

            # show le crème
            viewer.show()

    # report statistics
    opt_times = [t * 1000.0 for t in opt_times]  # Convert seconds to milliseconds
    print(f"Inference time over {num_predictions} samples (ms): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")

    labels = loss_terms_batch[0].keys()
    for label in labels:
        errors = [terms[label].item() for terms in loss_terms_batch]
        print(f"{label.capitalize()} over {num_predictions} samples: {mean(errors):.4f} (+-{stdev(errors):.4f})")

    if save_metrics:
        # Export force densities
        filename = f"{model_name}_{task_name}_q_eval.txt"
        filepath = os.path.join(DATA, filename)

        metrics = [f"{_q}\n" for _q in qs]
        with open(filepath, 'w') as output:
            output.writelines(metrics)

        print(f"Saved batch qs to {filepath}")


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(predict_batch)
