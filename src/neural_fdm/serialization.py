import equinox as eqx


def save_model(filename, model):
    """
    Serialize and save a model to a file.

    Parameters
    ----------
    filename: `str`
        The name of the file to save the model to.
        The file extension must be `.eqx`.
    model: `eqx.Module`
        The model to save.
    """
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(filename, model_skeleton):
    """
    Load a serialized model from a file.

    Parameters
    ----------
    filename: `str`
        The name of the file to load the model from.
        The file extension must be `.eqx`.
    model_skeleton: `eqx.Module`
        The reference skeleton of the model to load the model into.

    Returns
    -------
    model: `eqx.Module`
        The loaded model.
    """
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model_skeleton)
