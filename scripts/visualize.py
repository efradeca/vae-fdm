"""
Visualize the predictions of a model on a target shape.
"""
import os
from math import fabs
import yaml

import matplotlib.cm as cm
import numpy as np

import jax
import jax.numpy as jnp

from jax import vmap

import jax.random as jrn

from compas.colors import Color
from compas.colors import ColorMap
from compas.datastructures import mesh_transformed
from compas.geometry import Line
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import datastructure_updated

from jax_fdm.visualization import Viewer
from jax_fdm.visualization import Plotter

from neural_fdm import DATA
from neural_fdm import FIGURES

from neural_fdm.builders import build_loss_function
from neural_fdm.builders import build_mesh_from_generator
from neural_fdm.builders import build_data_generator
from neural_fdm.builders import build_connectivity_structure_from_generator
from neural_fdm.builders import build_neural_model

from neural_fdm.losses import print_loss_summary

from neural_fdm.serialization import load_model

from camera import CAMERA_CONFIG_BEZIER
from camera import CAMERA_CONFIG_BEZIER_TOP
from camera import CAMERA_CONFIG_TOWER

from shapes import BEZIERS
from shapes import TOWERS


# ===============================================================================
# Script function
# ===============================================================================

def visualize(
        model_names,
        task_name,
        shape_name=None,
        shape_index=0,
        seed=None,
        batch_size=None,
        view=True,
        use_camera_top=False,
        plot=False,
        save=False,
        edgewidth=(0.01, 0.25),
        edgecolor=None,
        show_reactions=True,
        reactionscale=1.0,
        plot_prediction=False,
        plot_target=True,
        plot_metric=None,
):
    """
    Visualize model predictions for one target shape.

    Parameters
    ----------
    model_names: `str`
        A list with the model names whose predictions you want to visualize.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    shape_name: `str` or `None`, optional
        The name of the shape to visualize.
        Supported shell shapes are pillow, dome, saddle, hypar, pringle, and cannon;
        and require of a `bezier_symmetric_double` generator.
        Supported tower shapes are either named by an integer or a float scalar.
        If the name is an integer, the generator should be `tower_ellipse`, and `tower_circle` if the name is a float.
        In general, if a name is provided, the optimization is performed on this shape, ignoring the batch.
    shape_index: `int`, optional
        The index of the shape to visualize in the batch.
    seed: `int` or `None`, optional
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`, optional
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    view: `bool`, optional
        If `True`, view the predicted shapes.
    use_camera_top: `bool`, optional
        If `True`, use the top view camera configuration.
    plot: `bool`, optional
        If `True`, plot the predicted shapes.
    save: `bool`, optional
        If `True`, save the plots.
    edgewidth: `tuple`, optional
        The width range of the edges.
    edgecolor: `str` or `None`, optional
        The color palette for the edges.
        Supported color palettes are fd to display force densities, and force to show forces.
        If `None`, the edges are colored by the force density in the shells tasks, and by the force in the tower tasks.
    show_reactions: `bool`, optional
        If `True`, show the reactions.
    reactionscale: `float`, optional
        The scale of the reactions.
    plot_prediction: `bool`, optional
        If `True`, plot the predicted shape.
    plot_target: `bool`, optional
        If `True`, plot the target shape.
    plot_metric: `str`, optional
        The name of the metric to plot.
        Supported metrics are forces, residuals, and deltas.        
    """
    if edgecolor is None:
        if task_name == "bezier":
            EDGECOLOR = "fd"
        elif task_name == "tower":
            EDGECOLOR = "force"
    else:
        EDGECOLOR = edgecolor

    PLOT_MODE = plot_metric
    EDGEWIDTH = edgewidth

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    seed = seed or config["seed"]
    training_params = config["training"]
    batch_size = batch_size or training_params["batch_size"]

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
    print(f"Making predictions on {generator_name} dataset with {bounds_name} bounds\n")
    print(f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges")

# ===============================================================================
# Create target
# ===============================================================================

    # sample target points
    if shape_name is not None and "bezier" in task_name:
        transform = BEZIERS[shape_name]
        transform = jnp.array(transform)
        xyz = generator.evaluate_points(transform)
    elif shape_name is not None and "tower" in task_name and bounds_name == "twisted":
        transforms = TOWERS[shape_name]
        transform = [jnp.array(T) for T in transforms]
        xyz = generator.evaluate_points(transform)
    # sample target points at random if no shape name
    else:
        xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))
        xyz = xyz_batch[shape_index, :]

    # create target mesh
    mesh_target = mesh.copy()
    _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
    for idx, key in mesh.index_key().items():
        mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

# ===============================================================================
# Load models
# ===============================================================================

    print("\nLoading models")
    if isinstance(model_names, str):
        model_names = [model_names]
    models_data = {model_name: {} for model_name in model_names}

    for model_name in model_names:
        filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
        _model_name = model_name.split("_")[0]
        model_skeleton = build_neural_model(_model_name, config, generator, model_key)
        model = load_model(filepath, model_skeleton)
        models_data[model_name]["model"] = model

# ===============================================================================
# Predictions
# ===============================================================================

    for model_name, data in models_data.items():
        print(f"\nPredicting with {model_name}")

        # get model
        model = data["model"]

        # calculate loss
        _, loss_terms = compute_loss(
            model,
            structure,
            xyz[None, :],
            aux_data=True
        )
        print_loss_summary(loss_terms)

        # predict equilibrium states for viz and i/o
        eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)

        # assemble datastructure for post-processing
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)
        network_hat.print_stats()
        print()

        # store datastructures
        models_data[model_name]["mesh"] = mesh_hat
        models_data[model_name]["network"] = network_hat

# ===============================================================================
# Predictions
# ===============================================================================

    deltas_all = []
    residuals_all = []
    forces_all = []
    forces_comp_all = []
    forces_tens_all = []
    areas_all = []

    for _, data in models_data.items():

        mesh = data["mesh"]
        network = data["network"]

        area = mesh.area()
        areas_all.append(area)

        for vkey in mesh.vertices():
            _xyz_target = mesh_target.vertex_coordinates(vkey)
            _xyz = mesh.vertex_coordinates(vkey)
            delta = distance_point_point(_xyz_target, _xyz)
            deltas_all.append(delta)

        for vkey in mesh.vertices():
            if mesh.is_vertex_on_boundary(vkey):
                continue
            residual = mesh.vertex_attributes(vkey, ["rx", "ry", "rz"])
            residuals_all.append(length_vector(residual))

        for edge in network.edges():

            force = network.edge_force(edge)
            force_abs = fabs(force)
            forces_all.append(force_abs)
            if force < 0.0:
                forces_comp_all.append(force_abs)
            else:
                forces_tens_all.append(force_abs)

    delta_min = 0.0
    delta_max = max(deltas_all)

    residual_min = 0.0
    residual_max = max(residuals_all)

    fmin = 0.0
    fmax = max(forces_all)

    fmin_comp = 0.0
    fmax_comp = max(forces_comp_all)

    fmin_tens = 0.0
    fmax_tens = max(forces_tens_all)

# ===============================================================================
# Viewing
# ===============================================================================

    # visualization
    if view:
        # pick camera configuration for task
        if task_name == "bezier":
            _width = 900
            CAMERA_CONFIG = CAMERA_CONFIG_BEZIER
            if use_camera_top:
                CAMERA_CONFIG = CAMERA_CONFIG_BEZIER_TOP
        elif task_name == "tower":
            _width = 450
            CAMERA_CONFIG = CAMERA_CONFIG_TOWER

        MESH_TARGET_OPACITY = 0.7
        MESH_TARGET_COLOR = Color.grey().lightened(100)

        # view target mesh alone
        if task_name == "bezier":
            viewer = Viewer(
                width=_width,
                height=900,
                show_grid=False,
                viewmode="lighted"
            )

            # modify view
            if use_camera_top:
                viewer.view.camera.view.current = viewer.view.camera.view.TOP

            viewer.view.camera.position = CAMERA_CONFIG["position"]
            viewer.view.camera.target = CAMERA_CONFIG["target"]
            viewer.view.camera.distance = CAMERA_CONFIG["distance"]

            _rotation = CAMERA_CONFIG.get("rotation")
            if _rotation:
                viewer.view.camera.rotation = _rotation

            # target shape
            print("\nViewing target...")
            MESH_TARGET_COLOR = Color.grey().lightened(100)
            MESH_TARGET_OPACITY = 0.7
            if use_camera_top:
                MESH_TARGET_OPACITY = 0.4
            viewer.add(
                 mesh_target,
                 show_points=False,
                 show_edges=False,
                 opacity=MESH_TARGET_OPACITY,
                 color=MESH_TARGET_COLOR,
            )

            if not use_camera_top:
                viewer.add(
                    FDNetwork.from_mesh(mesh_target),
                    as_wireframe=True,
                    show_points=False,
                    linewidth=4.0,  # 4.0 for 3d, 1.0 for top
                    color=Color.black().lightened()
                    )

            viewer.show()

        # view each model prediction
        for model_name, data in models_data.items():

            print(f"\nViewing {model_name} prediction...")
            viewer = Viewer(
                width=_width,
                height=900,
                show_grid=False,
                viewmode="lighted"
            )

            # modify view
            if use_camera_top:
                viewer.view.camera.view.current = viewer.view.camera.view.TOP

            viewer.view.camera.position = CAMERA_CONFIG["position"]
            viewer.view.camera.target = CAMERA_CONFIG["target"]
            viewer.view.camera.distance = CAMERA_CONFIG["distance"]

            _rotation = CAMERA_CONFIG.get("rotation")
            if _rotation:
                viewer.view.camera.rotation = _rotation

            # query datastructures
            network_hat = data["network"]
            mesh_hat = data["mesh"]

            # vertices to view
            vertices_2_view = list(mesh.vertices())
            if EDGECOLOR == "fd" and task_name == "bezier":
                vertices_2_view = [vkey for vkey in vertices_2_view if not mesh.is_vertex_on_boundary(vkey)]
            elif EDGECOLOR == "force" and task_name == "tower":
                vertices_2_view = [vkey for vkey in vertices_2_view if not mesh.is_vertex_on_boundary(vkey)]

            # edges to view
            # NOTE: we are not visualizing edges on boundaries since they are supported
            edges_2_view = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]

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

            # reaction view
            _reactionscale = reactionscale
            if task_name == "bezier":
                if model_name == "autoencoder":
                    _reactionscale *= 1.0  # 0.5
                elif model_name == "autoencoder_pinn":
                    _reactionscale *= -10.0
            else:
                if model_name == "autoencoder_pinn":
                    _reactionscale *= 20.0

            reactioncolor = Color.from_rgb255(0, 150, 10)  # load green
            if EDGECOLOR == "fd":
                reactioncolor = Color.grey().darkened()  # load dark gray

            if task_name == "bezier":

                _reactioncolor = {}
                for vkey in mesh.vertices():
                    _color = Color.pink()
                    if mesh.is_vertex_on_boundary(vkey):
                        _color = reactioncolor
                    _reactioncolor[vkey] = _color
                reactioncolor = _reactioncolor

            elif task_name == "tower":

                _reactioncolor = {}
                for vkey in mesh.vertices():
                    _color = Color.from_rgb255(0, 150, 10)
                    if mesh.is_vertex_on_boundary(vkey):
                        _color = reactioncolor
                    _reactioncolor[vkey] = _color
                reactioncolor = _reactioncolor

            # display stylized network
            viewer.add(
                network_hat,
                edgewidth=edgewidth,
                edgecolor=edgecolor,
                show_edges=True,
                edges=edges_2_view,
                nodes=vertices_2_view,
                show_loads=False,
                loadscale=1.0,
                show_reactions=show_reactions,
                reactionscale=_reactionscale,
                reactioncolor=reactioncolor
            )

            viewer.add(
                mesh_hat,
                show_points=False,
                show_edges=False,
                opacity=MESH_TARGET_OPACITY,
                color=MESH_TARGET_COLOR,
            )

            for _vertices in mesh.vertices_on_boundaries():
                viewer.add(
                    Polyline([mesh_hat.vertex_coordinates(vkey) for vkey in _vertices]),
                    linewidth=4.0,
                    color=Color.black().lightened()
                    )

            if task_name == "tower":
                rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring in rings:
                    ring = ring.tolist()
                    polygon = Polygon(ring)

                    viewer.add(polygon, opacity=0.5)
                    viewer.add(
                        Polyline(ring + ring[:1]),
                        linewidth=4.0,
                        color=Color.black().lightened()
                    )

            # show la crème
            viewer.show()

# ===============================================================================
# Plotting
# ===============================================================================

    if plot:

        nodes_to_plot = [node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2]
        edges_to_plot = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]
        nodesize = 12.0

        for name, data in models_data.items():
            print(f"\nPlotting for {name}")

            plotter = Plotter(figsize=(6, 6), dpi=150)

            # Transformations
            T = np.eye(4)
            if view:
                P = viewer.view.camera.projection(viewer.width, viewer.height)
                W = viewer.view.camera.viewworld()
                P[1, 1] = P[0, 0]
                T = P @ W

                nodesize = nodesize / 10.0

            if plot_prediction:
                mesh = mesh_transformed(mesh, T)
                plotter.add(mesh,
                            show_vertices=True,
                            vertexsize=nodesize,
                            vertexcolor={vkey: Color.white() for vkey in mesh.vertices()},
                            edgecolor={edge: Color(0.1, 0.1, 0.1) for edge in mesh.edges()},
                            edgewidth={edge: 1.0 for edge in mesh.edges()},
                            show_edges=True,
                            facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})

            if plot_target:

                mesh_target = mesh_transformed(mesh_target, T)

                for u, v in mesh_target.edges():
                    if mesh.is_edge_fully_supported((u, v)):
                        lc = Color(0.0, 0.0, 0.0)
                        ls = "solid"
                    else:
                        lc = Color(0.2, 0.2, 0.2)
                        ls = (0, (5, 3))  # "dotted", "dashed"
                    a, b = mesh_target.edge_coordinates(u, v)
                    line = Line(a, b)

                    plotter.add(line,
                                draw_as_segment=True,
                                color=lc,
                                linestyle=ls,
                                linewidth=1.0,
                                zorder=1300,
                                )

            # zoom in
            plotter.zoom_extents()

            if save:
                shape_id = shape_name or str(shape_index)
                parts = [generator_name, bounds_name, shape_id]
                if generator_name == "bezier_lerp":
                    parts.insert(1, str(int(10.0 * config['generator']['lerp_factor'])))
                if plot_prediction:
                    parts.append(name)
                if plot_target:
                    parts.append("target")
                filename = f"plot_{'_'.join(parts)}.pdf"
                FILE_OUT = os.path.abspath(os.path.join(FIGURES, filename))
                print(f"Saving plot to {FILE_OUT}")
                plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

            # show le crème
            plotter.show()

            if PLOT_MODE is None:
                continue

            print(f"\nPlotting {PLOT_MODE} for {name}")

            plotter = Plotter(
                figsize=(9, 9),
                dpi=150
            )

            # node work
            network_hat = data["network"]
            mesh_hat = data["mesh"]

            # plot sheets
            nodecolor = Color.white()
            if PLOT_MODE == "fd":

                edgecolor = PLOT_MODE                
                edges_to_plot = list(mesh.edges())
                edgewidth = (1.0, 9.0)

                show_reactions = False
                reaction_color = Color.pink()
                reaction_scale = 2.0
                if name == "autoencoder":
                    reaction_scale = 0.1

                plotter.add(mesh_hat,
                            show_vertices=False,
                            show_edges=False,
                            facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})

            elif PLOT_MODE == "residual":

                edgecolor = Color(0.1, 0.1, 0.1)
                edgewidth = 1.0

                edges_to_plot = list(mesh.edges())

                show_reactions = False
                reaction_color = Color.pink()
                reaction_scale = 0.5

                # cmap = cm.Oranges_r
                cmap = cm.RdPu
                vmin = residual_min
                vmax = residual_max

                residuals = []
                for vkey in mesh.vertices():
                    residual_vector = mesh_hat.vertex_attributes(vkey, ["rx", "ry", "rz"])
                    residual = length_vector(residual_vector)
                    if mesh.is_vertex_on_boundary(vkey):
                        residual = 0.0
                    residuals.append(residual)

                scales = remap_values(
                    residuals,
                    original_min=residual_min,
                    original_max=residual_max
                )

                z = residuals

                nodecolor = {}
                for key, scale in zip(mesh.vertices(), scales):
                    nodecolor[key] = cmap(scale)

            # delta plotting
            elif PLOT_MODE == "delta":

                edgecolor = Color(0.1, 0.1, 0.1)
                cmap = cm.turbo
                edgewidth = 1.0
                show_reactions = False
                reaction_color = None
                reaction_scale = 0.5

                deltas = []
                for vkey in mesh.vertices():
                    xyz_target = mesh_target.vertex_coordinates(vkey)
                    xyz = mesh_hat.vertex_coordinates(vkey)
                    delta = distance_point_point(xyz_target, xyz)
                    deltas.append(delta)

                scales = remap_values(
                    deltas,
                    original_min=delta_min,
                    original_max=delta_max
                )

                vmin = delta_min
                vmax = delta_max
                z = deltas

            if PLOT_MODE in ("delta", "residual"):
                # mesh grid
                x = np.array(mesh.vertices_attribute("x"))
                y = np.array(mesh.vertices_attribute("y"))
                X = np.reshape(x, (10, 10))
                Y = np.reshape(y, (10, 10))
                Z = np.reshape(np.array(z), (10, 10))

                nodecolor = {}
                for key, scale in zip(mesh.vertices(), scales):
                    nodecolor[key] = cmap(scale)

                plotter.axes.pcolormesh(
                    X,
                    Y,
                    Z,
                    vmin=vmin,
                    vmax=vmax,
                    shading='gouraud',
                    cmap=cmap,
                    zorder=1,
                )

            # query datastructures
            plotter.add(
                network_hat,
                edgewidth=edgewidth,
                edgecolor=edgecolor,
                show_edges=True,
                edges=edges_to_plot,
                show_nodes=True,
                nodes=nodes_to_plot,
                nodesize=nodesize,
                nodecolor=nodecolor,
                show_loads=False,
                loadscale=0.5,
                show_reactions=show_reactions,
                reactionscale=reaction_scale,
                reactioncolor=reaction_color,
            )

            # zoom in
            plotter.zoom_extents()

            if save:
                parts = [task_name, shape_name, PLOT_MODE, name]
                filename = f"plot_{'_'.join(parts)}.png"
                FILE_OUT = os.path.abspath(os.path.join(FIGURES, filename))
                print(f"Saving plot to {FILE_OUT}")
                plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

            # show le crème
            plotter.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(visualize)
