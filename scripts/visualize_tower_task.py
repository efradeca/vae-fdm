"""
Generate diagrams for the cablenet tower prediction task.
"""
import yaml

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import Plane

from jax_fdm.visualization import Viewer

from neural_fdm.builders import build_data_generator

from neural_fdm.generators import points_on_ellipse

from camera import CAMERA_CONFIG_TOWER as CAMERA_CONFIG


# ===============================================================================
# Script function
# ===============================================================================

def view_tower_task(seed=None, batch_size=None, shape_index=0):
    """
    View the description of the cablenet tower prediction task for a target shape.

    Parameters
    ----------
    seed: `int` or `None`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    shape_index: `int`
        The index of the shape to view.        
    """
    # pick camera configuration for task
    task_name = "tower"    
    _width = 450

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    if batch_size is None:
        training_params = config["training"]
        batch_size = training_params["batch_size"]

    # randomness
    key = jrn.PRNGKey(seed)
    _, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))
    xyz = xyz_batch[shape_index, :]

    # view task

    # create viewer
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

    # draw rings
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

    # draw planes, transparent, thick-ish boundary
    heights = jnp.linspace(0.0, generator.height, generator.num_levels)        
    
    for i, height in enumerate(heights):

        plane = Plane([0.0, 0.0, height], [0.0, 0.0, 1.0])

        circle = points_on_ellipse(
            generator.radius,
            generator.radius,
            height,
            generator.num_sides
            )
        circle = circle.tolist()

        if i in generator.levels_rings_comp:
            viewer.add(
                Polyline(circle + circle[:1]),
                linewidth=2.0,
                color=Color.grey().lightened()
                )
            # skip plane drawing for compression rings to avoid overlap
            continue

        viewer.add(
            plane,
            size=1.0,
            linewidth=0.1,
            color=Color.grey().lightened(10),
            opacity=0.1)

    # show viewer
    viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(view_tower_task)
