"""Tests for classical optimization baseline.

Verifies that direct optimization of force densities via L-BFGS-B
produces valid equilibrium solutions.
"""

import os
import sys

import jax.numpy as jnp
import jax.random as jrn
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def shell_setup():
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_mesh_from_generator,
    )

    config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    gen = build_data_generator(config)
    mesh = build_mesh_from_generator(config, gen)
    st = build_connectivity_structure_from_generator(config, gen)
    xyz = gen(jrn.PRNGKey(0))
    return config, mesh, st, xyz


class TestClassicalBaseline:
    def test_solves_without_error(self, shell_setup):
        """Classical solver runs and returns a result dict."""
        from neural_fdm.classical import solve_classical

        config, mesh, st, xyz = shell_setup
        result = solve_classical(xyz, config, mesh, st, maxiter=20, key=jrn.PRNGKey(42))
        assert "l_shape" in result
        assert "l_physics" in result
        assert "q_opt" in result

    def test_solution_is_finite(self, shell_setup):
        """All outputs are finite (no NaN/Inf)."""
        from neural_fdm.classical import solve_classical

        config, mesh, st, xyz = shell_setup
        result = solve_classical(xyz, config, mesh, st, maxiter=20, key=jrn.PRNGKey(42))
        assert jnp.all(jnp.isfinite(result["q_opt"]))
        assert jnp.all(jnp.isfinite(jnp.array(result["xyz_opt"])))
        assert jnp.isfinite(result["l_shape"])
        assert jnp.isfinite(result["l_physics"])

    def test_compression_only(self, shell_setup):
        """Shell task produces compression-only force densities (q <= 0)."""
        from neural_fdm.classical import solve_classical

        config, mesh, st, xyz = shell_setup
        result = solve_classical(xyz, config, mesh, st, maxiter=50, key=jrn.PRNGKey(42))
        assert float(jnp.max(result["q_opt"])) <= 0.01

    def test_l_physics_near_zero(self, shell_setup):
        """Equilibrium residual is near machine precision."""
        from neural_fdm.classical import solve_classical

        config, mesh, st, xyz = shell_setup
        result = solve_classical(xyz, config, mesh, st, maxiter=50, key=jrn.PRNGKey(42))
        assert result["l_physics"] < 1e-6

    def test_warm_start(self, shell_setup):
        """Warm-start with initial_q produces valid result."""
        from neural_fdm.classical import solve_classical

        config, mesh, st, xyz = shell_setup
        q_init = -jnp.ones(mesh.number_of_edges()) * 0.5
        result = solve_classical(xyz, config, mesh, st, maxiter=20, initial_q=q_init)
        assert jnp.all(jnp.isfinite(result["q_opt"]))
