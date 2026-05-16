import hashlib
import json
import os
import time
from datetime import datetime, timezone

import equinox as eqx


def _arch_hash(model):
    """Hash of the pytree structure (class names + leaf shapes) of a model.

    Used as a lightweight architecture fingerprint for `.eqx` metadata so
    a v1.0 weight file cannot be silently loaded with a v1.1 skeleton.
    """
    import jax.tree_util as jtu
    leaves, treedef = jtu.tree_flatten(model)
    structure = repr(treedef)
    leaf_shapes = []
    for leaf in leaves:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            leaf_shapes.append(f"{leaf.shape}:{leaf.dtype}")
        else:
            leaf_shapes.append(type(leaf).__name__)
    payload = structure + "|" + "|".join(leaf_shapes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_model(filename, model):
    """
    Serialize and save a model to a file.

    Writes a companion ``<filename>.meta.json`` with model class, version,
    architecture hash, and timestamp. The metadata is informational
    only -- ``load_model`` warns on mismatch but does not fail.

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

    try:
        import neural_fdm
        nf_version = getattr(neural_fdm, "__version__", "unknown")
    except Exception:
        nf_version = "unknown"
    meta = {
        "model_class": type(model).__name__,
        "module": type(model).__module__,
        "arch_hash": _arch_hash(model),
        "neural_fdm_version": nf_version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(filename + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(filename, model_skeleton):
    """
    Load a serialized model from a file.

    If a ``<filename>.meta.json`` companion exists, the skeleton's
    architecture hash is compared and a warning is printed on mismatch.
    Loading still proceeds (warn, not fail) so legacy weights without
    metadata stay usable.

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
    meta_path = filename + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            expected = meta.get("arch_hash")
            actual = _arch_hash(model_skeleton)
            if expected and expected != actual:
                import warnings
                warnings.warn(
                    f"Architecture hash mismatch loading {filename}: "
                    f"file was saved with hash {expected} but the provided "
                    f"skeleton has hash {actual}. Loading anyway -- predictions "
                    "may be wrong if the architecture has changed.",
                    RuntimeWarning,
                )
        except (json.JSONDecodeError, OSError):
            pass

    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model_skeleton)
