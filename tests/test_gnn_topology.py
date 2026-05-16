"""Tests for GNN encoder on different mesh sizes.

Verifies that the same GNN architecture can be instantiated for
different mesh resolutions (each as a separate model instance),
producing correct output shapes and finite predictions.
"""

import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def base_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "gnn_bezier.yml")
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_and_predict(config, num_uv, key):
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_neural_model,
    )
    config = dict(config)
    config["generator"] = dict(config["generator"])
    config["generator"]["num_uv"] = num_uv
    gen = build_data_generator(config)
    st = build_connectivity_structure_from_generator(config, gen)
    model = build_neural_model("formfinder", config, gen, key)
    x = gen(jrn.PRNGKey(0))
    out = model(x, st)
    return x, out, model


class TestGNNMeshSizes:
    def test_different_mesh_sizes(self, base_config):
        """GNN produces correct outputs for 6x6 and 10x10 meshes."""
        key = jrn.PRNGKey(42)
        x6, out6, _ = build_and_predict(base_config, 6, key)
        x10, out10, _ = build_and_predict(base_config, 10, key)
        assert x6.shape == (108,)   # 6*6*3
        assert x10.shape == (300,)  # 10*10*3
        assert out6.shape == (108,)
        assert out10.shape == (300,)

    def test_outputs_finite(self, base_config):
        """All outputs are finite for both mesh sizes."""
        key = jrn.PRNGKey(42)
        _, out6, _ = build_and_predict(base_config, 6, key)
        _, out10, _ = build_and_predict(base_config, 10, key)
        assert jnp.all(jnp.isfinite(out6))
        assert jnp.all(jnp.isfinite(out10))

    def test_gradients_flow(self, base_config):
        """Gradients flow through the full GNN encoder-decoder pipeline."""
        from neural_fdm.builders import (
            build_connectivity_structure_from_generator,
            build_data_generator,
            build_neural_model,
        )
        config = dict(base_config)
        config["generator"] = dict(config["generator"])
        config["generator"]["num_uv"] = 6
        gen = build_data_generator(config)
        st = build_connectivity_structure_from_generator(config, gen)
        key = jrn.PRNGKey(42)
        model = build_neural_model("formfinder", config, gen, key)
        x = gen(jrn.PRNGKey(0))

        def loss_fn(model):
            return jnp.sum(model(x, st))

        import equinox as eqx
        grads = eqx.filter_grad(loss_fn)(model)
        grad_norms = jax.tree_util.tree_map(
            lambda g: jnp.sum(jnp.abs(g)) if hasattr(g, 'shape') else 0.0,
            grads
        )
        total = sum(jax.tree_util.tree_leaves(grad_norms))
        assert float(total) > 0, "Gradients should be non-zero"
