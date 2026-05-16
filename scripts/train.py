"""
Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.
"""

import os
import time
from functools import partial

import equinox as eqx
import jax
import jax.random as jrn
import jax.tree_util as jtu
import yaml
from jax import vmap

from neural_fdm import DATA
from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_loss_function,
    build_neural_model,
    build_optimizer,
)
from neural_fdm.plotting import plot_losses as plot_loss_curves
from neural_fdm.serialization import load_model
from neural_fdm.serialization import save_model as save_model_fn
from neural_fdm.training import train_model

# ===============================================================================
# Script function
# ===============================================================================

def train(
        model_name="formfinder",
        task_name="bezier",
        from_pretrained=False,
        checkpoint_every=None,
        plot_losses=True,
        save_model=True,
        save_losses=True,
):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ----------
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, piggy, and variational_formfinder.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    from_pretrained: `bool`, optional
        If `True`, train the model starting from a pretrained version.
    checkpoint_every: `int` or `None`, optional
        If not None, save a model every checkpoint steps.
    plot_losses: `bool`, optional
        If `True`, plot the loss curves.
    save_model: `bool`, optional
        If `True`, save the trained model.
    save_losses: `bool`, optional
        If `True`, save the loss histories as text files.
    """
    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # resolve model name for saving and checkpointing
    filename = f"{model_name}"
    loss_params = config["loss"]
    if loss_params["residual"]["include"] > 0 and model_name not in ("formfinder", "variational_formfinder"):
        filename += "_pinn"
    filename += f"_{task_name}"

    # pick callback
    callback = None
    if checkpoint_every:
        callback = partial(
            checkpoint_model,
            checkpoint_step=checkpoint_every,
            filename=filename
        )

    # train model
    trained_model, loss_history = train_model_from_config(
        model_name,
        config,
        from_pretrained,
        callback=callback
    )

    if plot_losses:
        print("\nPlotting loss curves")
        plot_loss_curves(loss_history, labels=["loss"])

    if save_model:
        print("\nSaving model")

        # save trained model
        filepath = os.path.join(DATA, f"{filename}.eqx")
        save_model_fn(filepath, trained_model)
        print(f"Saved model to {filepath}")

    if save_losses:
        # Write the loss history as a single CSV with one column per
        # component (compact, pandas-friendly), plus per-component .txt
        # files for backward compatibility with interactive_designer.py's
        # Training tab (which scans for losses_*.txt patterns).
        import csv as _csv
        labels = list(loss_history[0].keys())
        clean_labels = ["_".join(label.split()) for label in labels]

        csv_path = os.path.join(DATA, f"losses_{filename}.csv")
        with open(csv_path, "w", newline="") as file:
            writer = _csv.writer(file)
            writer.writerow(["step"] + clean_labels)
            for step, values in enumerate(loss_history):
                row = [step]
                for label in labels:
                    val = values[label]
                    row.append(float(val.item() if hasattr(val, "item") else val))
                writer.writerow(row)
        print(f"Saved loss history to {csv_path}")

        for label, clean in zip(labels, clean_labels):
            txt_path = os.path.join(DATA, f"losses_{filename}_{clean}.txt")
            with open(txt_path, "w") as file:
                for values in loss_history:
                    val = values[label]
                    file.write(f"{float(val.item() if hasattr(val, 'item') else val)}\n")


# ===============================================================================
# Train functions
# ===============================================================================

def train_model_from_config(model_name, config, pretrained=False, callback=None):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ----------
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, piggy, and variational_formfinder.
    config: `dict`
        A dictionary with the hyperparameters configuration.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    pretrained: `bool`
        If `True`, train the model starting from a pretrained version of it.
    callback: `Callable`
        A callback function to call at every train step.
    """
    # unpack parameters
    seed = config["seed"]
    training_params = config["training"]
    batch_size = training_params["batch_size"]
    steps = training_params["steps"]
    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create experiment
    print(f"\nTraining {model_name} on {generator_name} dataset with {bounds_name} bounds")
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    compute_loss = build_loss_function(config, generator)
    model = build_neural_model(model_name, config, generator, model_key)
    optimizer = build_optimizer(config)

    if pretrained:
        print("Starting from pretrained model")
        task_name = generator_name.split("_")[0]
        filepath = os.path.join(DATA, f"{model_name}_{task_name}_pretrain.eqx")
        model = load_model(filepath, model)

    # sample initial data batch
    xyz = vmap(generator)(jrn.split(generator_key, batch_size))

    # warmstart
    start_loss = compute_loss(model, structure, xyz)
    print(f"The structure has {structure.num_vertices} vertices and {structure.num_edges} edges")
    print(f"Model parameter count: {count_model_params(model)}")
    print(f"{model_name.capitalize()} start loss: {start_loss:.6f}")

    # train models
    print("\nTraining")
    start = time.perf_counter()
    train_data = train_model(
        model,
        structure,
        optimizer,
        generator,
        loss_fn=compute_loss,
        num_steps=steps,
        batch_size=batch_size,
        key=generator_key,
        callback=callback
        )
    end = time.perf_counter()

    print("\nTraining completed")
    print(f"Training time: {end - start:.4f} s")

    trained_model, loss_history = train_data

    # Report final loss from the loss history rather than a fresh,
    # unjitted compute_loss call. The unjitted path previously triggered
    # a JAX UnexpectedTracerError and is otherwise unnecessary -- the
    # loss history already contains the last step's value.
    if loss_history:
        last = loss_history[-1]
        last_loss = last.get("loss", last) if isinstance(last, dict) else last
        try:
            last_loss = float(last_loss)
            print(f"{model_name.capitalize()} last loss: {last_loss:.6f}")
        except (TypeError, ValueError):
            print(f"{model_name.capitalize()} last loss: {last_loss}")

    return train_data


# ===============================================================================
# Helper functions
# ===============================================================================

def checkpoint_model(
        model,
        opt_state,
        loss_vals,
        step,
        checkpoint_step,
        filename
):
    """
    Checkpoint a model. Function to be used as a callback in the training loop.

    Parameters
    ----------
    model: `eqx.Module`
        The model to checkpoint.
    opt_state: `eqx.Module`
        The optimizer state.
    loss_vals: `dict`
        The loss values.
    step: `int`
        The current training step.
    checkpoint_step: `int`
        The step interval at which to checkpoint the model.
    filename: `str`
        The filename to save the model to.
    """
    if step > 0 and step % checkpoint_step == 0:
        filepath = os.path.join(DATA, f"{filename}_{step}.eqx")
        save_model_fn(filepath, model)


def count_model_params(model):
    """
    Count the number of trainable model parameters.

    Parameters
    ----------
    model: `eqx.Module`
        The model to count the parameters of.

    Returns
    -------
    count: `int`
        The number of trainable model parameters.
    """
    spec = eqx.is_inexact_array

    return sum(x.size for x in jtu.tree_leaves(eqx.filter(model, spec)))


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(train)
