"""Tests for the public predict_equilibrium API.

Verifies that the numpy-based API works for both deterministic
(formfinder) and variational (VAE) models.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class TestPredictEquilibriumDeterministic:
    @pytest.fixture
    def det_result(self):
        from neural_fdm.interop.api import predict_equilibrium

        model_path = os.path.join(DATA, "formfinder_bezier.eqx")
        config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "bezier.yml")
        if not os.path.exists(model_path):
            pytest.skip("Model weights not available")
        return predict_equilibrium(
            vertices=np.zeros((100, 3)),
            model_path=model_path,
            config_path=config_path,
        )

    def test_returns_expected_keys(self, det_result):
        for key in ["vertices", "force_densities", "forces", "lengths", "residuals", "inference_time_ms"]:
            assert key in det_result

    def test_vertices_shape(self, det_result):
        assert det_result["vertices"].shape == (100, 3)

    def test_force_densities_shape(self, det_result):
        assert det_result["force_densities"].shape == (180,)

    def test_all_finite(self, det_result):
        assert np.all(np.isfinite(det_result["vertices"]))
        assert np.all(np.isfinite(det_result["force_densities"]))


class TestPredictEquilibriumVAE:
    @pytest.fixture
    def vae_result(self):
        from neural_fdm.interop.api import predict_equilibrium

        model_path = os.path.join(DATA, "variational_formfinder_variational_bezier.eqx")
        config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "variational_bezier.yml")
        if not os.path.exists(model_path):
            pytest.skip("VAE model weights not available")
        return predict_equilibrium(
            vertices=np.zeros((100, 3)),
            model_path=model_path,
            config_path=config_path,
        )

    def test_returns_expected_keys(self, vae_result):
        for key in ["vertices", "force_densities", "forces", "lengths", "residuals"]:
            assert key in vae_result

    def test_vertices_shape(self, vae_result):
        assert vae_result["vertices"].shape == (100, 3)

    def test_all_finite(self, vae_result):
        assert np.all(np.isfinite(vae_result["vertices"]))
        assert np.all(np.isfinite(vae_result["force_densities"]))


class TestPredictEquilibriumTower:
    @pytest.fixture
    def tower_result(self):
        from neural_fdm.interop.api import predict_equilibrium

        model_path = os.path.join(DATA, "formfinder_tower.eqx")
        config_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "tower.yml")
        if not os.path.exists(model_path):
            pytest.skip("Tower model weights not available")

        # Tower has 336 vertices (21 levels × 16 sides)
        return predict_equilibrium(
            vertices=np.zeros((336, 3)),
            model_path=model_path,
            config_path=config_path,
        )

    def test_returns_expected_keys(self, tower_result):
        for key in ["vertices", "force_densities", "forces", "lengths", "residuals"]:
            assert key in tower_result

    def test_vertices_shape(self, tower_result):
        assert tower_result["vertices"].shape == (336, 3)

    def test_all_finite(self, tower_result):
        assert np.all(np.isfinite(tower_result["vertices"]))
        assert np.all(np.isfinite(tower_result["force_densities"]))
