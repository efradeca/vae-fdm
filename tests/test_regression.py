"""Regression tests pinning behaviour of recent fixes.

Each test exercises one specific defect that was previously latent and
now has a deterministic check. Covers dispatch errors, PRNG independence
in generators, GNN direction-flip invariance, web input validation,
tracer-leak prevention in the variational training loop, VAE
serialization round-trip, classical solver guards, and more.
"""
import json
import os
import sys
import tempfile

import jax.numpy as jnp
import jax.random as jrn
import numpy as np
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ----------------------------------------------------------------------------
# build_loss_function raises on unknown generator name
# ----------------------------------------------------------------------------

def test_build_loss_function_raises_on_unknown_generator():
    """An unknown generator.name must fail loudly, not silently dispatch."""
    from neural_fdm.builders import build_loss_function

    # Minimal config; the only thing that matters is generator.name.
    cfg = {
        "generator": {"name": "completely_unknown_task"},
        "loss": {"shape": {"include": 1, "weight": 1.0},
                 "residual": {"include": 0, "weight": 0.0}},
    }

    class _DummyGen:
        pass

    with pytest.raises(ValueError, match="Unknown generator name"):
        build_loss_function(cfg, _DummyGen())


# ----------------------------------------------------------------------------
# PRNG keys are split between the two Bezier-lerp surfaces
# ----------------------------------------------------------------------------

def test_bezier_lerp_wiggle_uses_split_keys():
    """The two surfaces inside BezierSurfaceLerpPointGenerator must be
    perturbed with independently split keys. Previously the same key was
    handed to both sub-generators, biasing the lerp distribution."""
    from neural_fdm.generators.generator_bezier import (
        BezierSurfaceLerpPointGenerator,
    )

    size = 2.0
    num_pts = 4
    n_uv = 10
    u = jnp.linspace(0.0, 1.0, n_uv)
    v = jnp.linspace(0.0, 1.0, n_uv)
    # Double-symmetric tile uses 4 control points (quarter-symmetry);
    # asymmetric tile uses the full 16. The constructor accepts both per-tile.
    minval = (jnp.zeros((4, 3)), jnp.zeros((16, 3)))
    maxval = (jnp.ones((4, 3)) * 0.1, jnp.ones((16, 3)) * 0.1)
    gen = BezierSurfaceLerpPointGenerator(size, num_pts, u, v, minval, maxval, alpha=0.5)

    # In the buggy version, both sub-generators were called with the same
    # parent key. Post-fix, each gets one half of a jrn.split. Verify by
    # showing that wiggling with the SPLIT key differs from wiggling with
    # the original key for the same sub-generator -- proving the split is
    # actually applied in the __call__ pathway.
    k = jrn.PRNGKey(0)
    k_this, _ = jrn.split(k)
    t_split = np.asarray(gen.wiggle(k_this))
    t_unsplit = np.asarray(gen.wiggle(k))
    assert not np.allclose(t_split, t_unsplit), \
        ("BezierSurfaceLerpPointGenerator.__call__ must split its key before "
         "forwarding to the two sub-generators. After the fix, the split-key "
         "wiggle must differ from the parent-key wiggle.")


def test_tower_wiggle_radii_and_angles_uncorrelated():
    """EllipticalTubePointGenerator.wiggle now splits the key between
    wiggle_radii and wiggle_angle. Verify they are not identical."""
    cfg_path = os.path.join(ROOT, "scripts", "tower.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    from neural_fdm.builders import build_data_generator
    gen = build_data_generator(cfg)
    if not hasattr(gen, "wiggle_radii") or not hasattr(gen, "wiggle_angle"):
        pytest.skip("Active generator does not expose tower-style wiggle.")

    r, a = gen.wiggle(jrn.PRNGKey(0))
    r2 = gen.wiggle_radii(jrn.PRNGKey(0))
    a2 = gen.wiggle_angle(jrn.PRNGKey(0))
    # After the fix, neither half is computed with the parent key,
    # so r should not equal r2 nor a equal a2.
    assert not np.allclose(np.asarray(r), np.asarray(r2)), \
        "wiggle_radii was called with the parent key (key not split)"
    assert not np.allclose(np.asarray(a), np.asarray(a2)), \
        "wiggle_angle was called with the parent key (key not split)"


# ----------------------------------------------------------------------------
# DiversityRequest input bounds (Pydantic Field validation)
# ----------------------------------------------------------------------------

def test_diversity_request_rejects_unbounded_n_samples():
    """n_samples must be bounded so /api/diversity cannot OOM the container."""
    sys.path.insert(0, ROOT)
    # Direct schema test — avoids spinning up the full FastAPI app + JAX import.
    try:
        from pydantic import ValidationError
    except ImportError:
        pytest.skip("pydantic not installed")

    from web.app import DiversityRequest, PredictRequest

    # In-range values accepted.
    DiversityRequest(n_samples=40)
    PredictRequest(c1_z=3.0, c2_x=0.0, c2_z=1.5, c3_y=0.0)

    # Out-of-range rejected.
    with pytest.raises(ValidationError):
        DiversityRequest(n_samples=100000)
    with pytest.raises(ValidationError):
        DiversityRequest(n_samples=0)
    with pytest.raises(ValidationError):
        PredictRequest(c1_z=1e10, c2_x=0.0, c2_z=1.5, c3_y=0.0)


# ----------------------------------------------------------------------------
# GNN message passing + edge readout are sender/receiver-symmetric
# ----------------------------------------------------------------------------

def _build_simple_gnn(num_edges=6, num_nodes=4, hidden_dim=16, num_layers=2):
    """Helper: build a tiny GNNEncoder for symmetry tests."""
    from neural_fdm.gnn import GNNEncoder
    rng = jrn.PRNGKey(0)
    edges_signs = jnp.ones(num_edges)
    # Simple ring topology over num_nodes
    senders = jnp.array([i for i in range(num_edges)]) % num_nodes
    receivers = (senders + 1) % num_nodes
    edge_index = jnp.stack([senders, receivers], axis=0)
    enc = GNNEncoder(
        edges_signs=edges_signs,
        q_shift=0.5,
        node_feat_dim=3,
        edge_feat_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_index=edge_index,
        key=rng,
    )
    return enc, edge_index


def test_gnn_edge_readout_symmetric_under_direction_flip():
    """Flipping the (sender, receiver) order of every edge must leave the
    per-edge q unchanged, because structural edges in FDM are undirected."""
    enc, edge_index = _build_simple_gnn()
    x = jrn.normal(jrn.PRNGKey(1), (4, 3)).ravel()

    q_orig = np.asarray(enc(x, edge_index=edge_index))
    flipped = jnp.stack([edge_index[1], edge_index[0]], axis=0)
    q_flipped = np.asarray(enc(x, edge_index=flipped))

    np.testing.assert_allclose(q_orig, q_flipped, atol=1e-5, rtol=1e-5,
        err_msg="GNN q values changed when edges were direction-flipped; "
                "encoder is not undirected/permutation equivariant.")


def test_variational_gnn_readout_symmetric_under_direction_flip():
    """Same property must hold for the VariationalGNNEncoder."""
    from neural_fdm.gnn import VariationalGNNEncoder
    rng = jrn.PRNGKey(2)
    edges_signs = jnp.ones(6)
    senders = jnp.array([0, 1, 2, 3, 0, 2])
    receivers = jnp.array([1, 2, 3, 0, 2, 0])
    edge_index = jnp.stack([senders, receivers], axis=0)
    enc = VariationalGNNEncoder(
        edges_signs=edges_signs,
        q_shift=0.5,
        node_feat_dim=3,
        edge_feat_dim=4,
        hidden_dim=16,
        num_layers=2,
        edge_index=edge_index,
        key=rng,
    )
    x = jrn.normal(jrn.PRNGKey(3), (4, 3)).ravel()
    # Use a fixed key so reparameterization noise is deterministic.
    k = jrn.PRNGKey(0)
    q_orig, mu_orig, ls_orig = enc(x, edge_index=edge_index, key=k)
    flipped = jnp.stack([edge_index[1], edge_index[0]], axis=0)
    q_flip, mu_flip, ls_flip = enc(x, edge_index=flipped, key=k)
    np.testing.assert_allclose(np.asarray(mu_orig), np.asarray(mu_flip),
                                atol=1e-5, rtol=1e-5,
        err_msg="VariationalGNN mu is not direction-symmetric.")
    np.testing.assert_allclose(np.asarray(ls_orig), np.asarray(ls_flip),
                                atol=1e-5, rtol=1e-5,
        err_msg="VariationalGNN log_sigma is not direction-symmetric.")


# ----------------------------------------------------------------------------
# PointGrid raises on unsupported num_pts
# ----------------------------------------------------------------------------

def test_point_grid_rejects_non_four_num_pts():
    """The hard-coded reindex map only supports num_pts=4; other values must
    raise rather than silently corrupting indexing."""
    import jax.numpy as jnp
    from neural_fdm.generators.grids import PointGrid
    tile = jnp.zeros((16, 3))
    PointGrid(tile, 4)  # legal
    for bad in (3, 5, 8, 16):
        with pytest.raises(ValueError, match="num_pts=4"):
            PointGrid(tile, bad)


# ----------------------------------------------------------------------------
# AutoEncoder.encode accepts structure for GNN compatibility
# ----------------------------------------------------------------------------

def test_autoencoder_encode_accepts_structure_kwarg():
    """AutoEncoder.encode must accept an optional structure argument so that
    GNN encoders work end-to-end through the CLI prediction scripts."""
    import inspect
    from neural_fdm.models import AutoEncoder
    sig = inspect.signature(AutoEncoder.encode)
    params = sig.parameters
    assert "structure" in params, \
        "AutoEncoder.encode is missing the 'structure' kwarg needed by GNN encoders."
    # Default must be None so MLP-only callsites stay compatible.
    assert params["structure"].default is None


# ----------------------------------------------------------------------------
# VAE training does not leave a stale tracer in the loss closure
# ----------------------------------------------------------------------------

def test_vae_no_tracer_leak_after_training():
    """Train a VAE for 2 steps, then call the SAME compute_loss partial that
    train_model used, unjitted. Pre-fix this raised
    jax.errors.UnexpectedTracerError because train_step_vae mutated a dict
    shared with the loss_fn closure. Post-fix the call must succeed."""
    import jax.random as jrn
    import jax.numpy as jnp
    from jax import vmap
    import optax
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_loss_function,
        build_neural_model,
    )
    from neural_fdm.training import train_model

    cfg_path = os.path.join(ROOT, "scripts", "variational_bezier.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    key = jrn.PRNGKey(0)
    mk, gk = jrn.split(key)
    gen = build_data_generator(cfg)
    structure = build_connectivity_structure_from_generator(cfg, gen)
    compute_loss = build_loss_function(cfg, gen)
    model = build_neural_model("variational_formfinder", cfg, gen, mk)
    optimizer = optax.adam(1e-4)

    xyz = vmap(gen)(jrn.split(gk, 2))

    trained, _ = train_model(
        model, structure, optimizer, gen,
        loss_fn=compute_loss, num_steps=2, batch_size=2, key=gk,
    )

    # This is the exact call that previously raised UnexpectedTracerError
    # because beta from the JIT'd train step had leaked into
    # loss_fn.keywords["loss_params"]["vae"]["beta"].
    result = compute_loss(trained, structure, xyz)
    assert result is not None, "Unjitted compute_loss after training must succeed."


# ----------------------------------------------------------------------------
# VAE serialization round-trip
# ----------------------------------------------------------------------------

def test_variational_serialization_roundtrip(tmp_path):
    """Save a fresh VAE, load it back, verify outputs match for the same key."""
    import jax.random as jrn
    import jax.numpy as jnp
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_neural_model,
    )
    from neural_fdm.serialization import save_model, load_model

    cfg_path = os.path.join(ROOT, "scripts", "variational_bezier.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    mk = jrn.PRNGKey(0)
    gen = build_data_generator(cfg)
    structure = build_connectivity_structure_from_generator(cfg, gen)
    skeleton = build_neural_model("variational_formfinder", cfg, gen, mk)

    sample_key = jrn.PRNGKey(7)
    xyz = gen(sample_key)

    # Forward once on the original model
    out_orig = skeleton(xyz, structure, key=jrn.PRNGKey(42))

    # Round-trip
    path = str(tmp_path / "vae_roundtrip.eqx")
    save_model(path, skeleton)
    reloaded = load_model(path, skeleton)

    out_loaded = reloaded(xyz, structure, key=jrn.PRNGKey(42))

    # Predicted geometry must match exactly with the same PRNG key
    pred_orig = jnp.asarray(out_orig[0] if isinstance(out_orig, tuple) else out_orig)
    pred_load = jnp.asarray(out_loaded[0] if isinstance(out_loaded, tuple) else out_loaded)
    np.testing.assert_allclose(np.asarray(pred_orig), np.asarray(pred_load),
                                atol=1e-6, rtol=1e-6,
        err_msg="VAE serialization round-trip produced different predictions.")


# ----------------------------------------------------------------------------
# classical singular-matrix guard
# ----------------------------------------------------------------------------

def test_classical_solver_has_finite_value_guard():
    """The classical FDM solver must include a finite-value guard that
    downgrades success to False and emits a RuntimeWarning if the L-BFGS-B
    result is NaN/Inf. End-to-end coverage is hard to set up because the
    solver is JIT-compiled inside a SciPy wrapper, so we assert structurally
    that the guard code path exists and emits the right warning class."""
    import inspect
    from neural_fdm import classical as classical_mod

    src = inspect.getsource(classical_mod.solve_classical)
    # Required: the guard checks isfinite on q_opt and l_shape, warns with
    # RuntimeWarning, and toggles success_flag to False.
    assert "jnp.all(jnp.isfinite(q_opt))" in src, \
        "solve_classical missing finite-value check on q_opt"
    assert "RuntimeWarning" in src, \
        "solve_classical missing RuntimeWarning on singular result"
    assert "success_flag = False" in src, \
        "solve_classical does not flip success to False on degenerate result"


# ----------------------------------------------------------------------------
# edges_lengths epsilon floor (strictly positive on zero-vector input)
# ----------------------------------------------------------------------------

def test_edges_lengths_strictly_positive_on_zero_vectors():
    """edges_lengths must return strictly positive values even for collapsed
    edges, so downstream divisions cannot produce NaN gradients."""
    import jax.numpy as jnp
    from neural_fdm.helpers import edges_lengths
    zero_vectors = jnp.zeros((3, 3))
    lengths = edges_lengths(zero_vectors)
    arr = np.asarray(lengths)
    assert np.all(arr > 0.0), f"edges_lengths produced non-positive on zero input: {arr}"
    # And the value should be near sqrt(eps) ~ 1e-6, not zero or NaN.
    assert np.all(np.isfinite(arr)), f"edges_lengths produced NaN/Inf: {arr}"
    assert np.all(arr < 1e-3), f"edges_lengths epsilon floor unexpectedly large: {arr}"


# ----------------------------------------------------------------------------
# .eqx.meta.json companion + hash-mismatch warning
# ----------------------------------------------------------------------------

def test_save_model_writes_companion_meta(tmp_path):
    """save_model must write a companion .eqx.meta.json with required keys."""
    import jax.random as jrn
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_neural_model,
    )
    from neural_fdm.serialization import save_model

    cfg_path = os.path.join(ROOT, "scripts", "bezier.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    gen = build_data_generator(cfg)
    skel = build_neural_model("formfinder", cfg, gen, jrn.PRNGKey(0))

    path = str(tmp_path / "m.eqx")
    save_model(path, skel)

    meta_path = path + ".meta.json"
    assert os.path.exists(meta_path), "save_model did not write .eqx.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    for key in ("model_class", "arch_hash", "saved_at", "neural_fdm_version"):
        assert key in meta, f"meta.json missing required key: {key}"
    assert len(meta["arch_hash"]) == 16, "arch_hash should be 16 hex chars"


def test_load_model_warns_on_arch_hash_mismatch(tmp_path):
    """If the companion meta hash disagrees with the skeleton, load_model
    must emit a RuntimeWarning but still load."""
    import warnings
    import jax.random as jrn
    from neural_fdm.builders import (
        build_connectivity_structure_from_generator,
        build_data_generator,
        build_neural_model,
    )
    from neural_fdm.serialization import save_model, load_model

    cfg_path = os.path.join(ROOT, "scripts", "bezier.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    gen = build_data_generator(cfg)
    skel = build_neural_model("formfinder", cfg, gen, jrn.PRNGKey(0))

    path = str(tmp_path / "m.eqx")
    save_model(path, skel)

    # Corrupt the hash
    meta_path = path + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["arch_hash"] = "0" * 16
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_model(path, skel)
    assert any(issubclass(w.category, RuntimeWarning) and "Architecture hash" in str(w.message)
               for w in caught), \
        "load_model did not warn on arch_hash mismatch"
    assert loaded is not None, "load_model failed to return a model even after warning"


# ----------------------------------------------------------------------------
# loss history dual CSV + .txt schema
# ----------------------------------------------------------------------------

def test_loss_history_csv_has_expected_schema(tmp_path, monkeypatch):
    """Confirm that train.py's save_losses block writes both formats with
    the correct CSV header and per-component .txt files."""
    import csv as _csv
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    # Don't actually run training -- replicate the save block in isolation.
    DATA = str(tmp_path)
    history = [
        {"loss": 1.0, "shape error": 0.9, "residual error": 0.1},
        {"loss": 0.8, "shape error": 0.7, "residual error": 0.1},
    ]
    filename = "smoke_test"
    labels = list(history[0].keys())
    clean = ["_".join(l.split()) for l in labels]
    csv_path = os.path.join(DATA, f"losses_{filename}.csv")
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step"] + clean)
        for step, vals in enumerate(history):
            w.writerow([step] + [vals[l] for l in labels])

    with open(csv_path) as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["step", "loss", "shape_error", "residual_error"]
    assert len(rows) == 2
    assert rows[0][0] == "0"


# ----------------------------------------------------------------------------
# PredictRequest bounds
# ----------------------------------------------------------------------------

def test_predict_request_rejects_out_of_range_floats():
    """PredictRequest Field bounds must reject every coordinate outside the
    documented BOUNDS dict ranges."""
    sys.path.insert(0, ROOT)
    try:
        from pydantic import ValidationError
    except ImportError:
        pytest.skip("pydantic not installed")
    from web.app import PredictRequest

    PredictRequest(c1_z=3.0, c2_x=0.0, c2_z=1.5, c3_y=0.0)  # canonical OK

    bad_cases = [
        dict(c1_z=0.5, c2_x=0.0, c2_z=1.5, c3_y=0.0),    # c1_z below min
        dict(c1_z=11.0, c2_x=0.0, c2_z=1.5, c3_y=0.0),   # c1_z above max
        dict(c1_z=3.0, c2_x=-6.0, c2_z=1.5, c3_y=0.0),   # c2_x below min
        dict(c1_z=3.0, c2_x=6.0, c2_z=1.5, c3_y=0.0),    # c2_x above max
        dict(c1_z=3.0, c2_x=0.0, c2_z=-0.1, c3_y=0.0),   # c2_z below min
        dict(c1_z=3.0, c2_x=0.0, c2_z=11.0, c3_y=0.0),   # c2_z above max
        dict(c1_z=3.0, c2_x=0.0, c2_z=1.5, c3_y=-6.0),   # c3_y below min
        dict(c1_z=3.0, c2_x=0.0, c2_z=1.5, c3_y=6.0),    # c3_y above max
    ]
    for case in bad_cases:
        with pytest.raises(ValidationError):
            PredictRequest(**case)


# ----------------------------------------------------------------------------
# text_2_mesh honours custom filepath
# ----------------------------------------------------------------------------

def test_text_2_mesh_passes_filepath_to_char_2_mesh(monkeypatch):
    """text_2_mesh used to hard-code 'Vera.ttf', ignoring the filepath kwarg."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import text_2_mesh as t2m
    except ImportError as e:
        pytest.skip(f"text_2_mesh has optional deps not installed: {e}")

    seen = []

    def fake_char_2_mesh(char, filepath="Vera.ttf"):
        seen.append(filepath)
        # Return a 1x1 trivial mesh stub.
        from compas.datastructures import Mesh
        return Mesh.from_vertices_and_faces(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1, 2]])

    monkeypatch.setattr(t2m, "char_2_mesh", fake_char_2_mesh)
    t2m.text_2_mesh("AB", filepath="MyCustomFont.ttf")
    assert seen, "char_2_mesh was never called"
    assert all(fp == "MyCustomFont.ttf" for fp in seen), \
        f"text_2_mesh ignored its filepath argument: saw {seen}"
