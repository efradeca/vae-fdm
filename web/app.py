"""FastAPI backend for VAE-FDM web explorer.

Serves the Three.js frontend and provides a /api/predict endpoint
that runs JAX inference on CPU.
"""
import os
import sys
import time

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root so neural_fdm is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax
import jax.numpy as jnp
import jax.random as jrn

from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_mesh_from_generator,
    build_neural_model,
)
from neural_fdm.helpers import edges_forces, edges_lengths, edges_vectors
from neural_fdm.serialization import load_model

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
TASK = "bezier"
SEED = 90
CFG_PATH = os.path.join(ROOT, "scripts", f"{TASK}.yml")
MODEL_PATH = os.path.join(ROOT, "data", f"formfinder_{TASK}.eqx")

with open(CFG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

key = jrn.PRNGKey(SEED)
mk, _ = jax.random.split(key, 2)

gen = build_data_generator(cfg)
structure = build_connectivity_structure_from_generator(cfg, gen)
mesh = build_mesh_from_generator(cfg, gen)

# Build and load model
skeleton = build_neural_model("formfinder", cfg, gen, mk)
model = load_model(MODEL_PATH, skeleton)

# JIT-compile predict function
@jax.jit
def _predict(x):
    xh, (q, xf, ld) = model(x, structure, aux_data=True)
    return xh, q, ld

# Warm up JIT
NU = cfg["generator"]["num_uv"]
_predict(jnp.zeros(NU * NU * 3))
print(f"Model loaded and JIT-compiled. Grid: {NU}x{NU}")

# Static topology (edges, boundary vertices, faces) - sent once
EDGES = np.array(list(mesh.edges())).tolist()
BOUNDARY = sorted(set(mesh.vertices_on_boundary()))
TILE = np.array(gen.surface.grid.tile).tolist()
SIZE = cfg["generator"]["size"]

# Saddle bounds from builders.py
BOUNDS = {
    "c1_z": {"min": 1.0, "max": 10.0, "default": 3.0, "label": "c1.z height"},
    "c2_x": {"min": -5.0, "max": 5.0, "default": 0.0, "label": "c2.x spread"},
    "c2_z": {"min": 0.0, "max": 10.0, "default": 1.5, "label": "c2.z edge"},
    "c3_y": {"min": -5.0, "max": 5.0, "default": 0.0, "label": "c3.y curve"},
}

# Preset shapes
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from shapes import BEZIERS
PRESETS = {}
for name, t in BEZIERS.items():
    PRESETS[name] = {
        "c1_z": t[0][2], "c2_x": t[1][0], "c2_z": t[1][2], "c3_y": t[2][1]
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="VAE-FDM Explorer")


class PredictRequest(BaseModel):
    c1_z: float = 3.0
    c2_x: float = 0.0
    c2_z: float = 1.5
    c3_y: float = 0.0


@app.get("/api/topology")
def get_topology():
    """Return static mesh topology (called once on page load)."""
    return {
        "edges": EDGES,
        "boundary": BOUNDARY,
        "num_vertices": NU * NU,
        "num_uv": NU,
        "tile": TILE,
        "bounds": BOUNDS,
        "presets": PRESETS,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Run neural FDM inference and return geometry + scalars."""
    transform = jnp.array([
        [0.0, 0.0, req.c1_z],
        [req.c2_x, 0.0, req.c2_z],
        [0.0, req.c3_y, 0.0],
        [0.0, 0.0, 0.0],
    ])

    t0 = time.perf_counter()

    # Target surface
    xyz_target = gen.evaluate_points(transform)
    target_np = np.array(xyz_target).reshape(-1, 3)

    # Neural prediction
    pred, q, ld = _predict(xyz_target)
    pred_np = np.array(pred).reshape(-1, 3)
    q_np = np.array(q).flatten()

    # Post-process
    xj = jnp.reshape(pred, (-1, 3))
    v = edges_vectors(xj, structure.connectivity)
    lengths = np.array(edges_lengths(v)).flatten()
    # F = q * L (element-wise, not the matrix version from edges_forces)
    forces = q_np * lengths

    dt = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "target": target_np.tolist(),
        "predicted": pred_np.tolist(),
        "q": q_np.tolist(),
        "forces": forces.tolist(),
        "lengths": lengths.tolist(),
        "inference_ms": round(dt, 2),
    })


# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
