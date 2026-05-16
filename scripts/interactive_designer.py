"""
Neural FDM - Interactive Form-Finding Explorer
===============================================
Open-source standalone reproducing Pastrana et al. (ICLR 2025), Figure 14.

Features:
  - Draggable 3D control points (like paper GIF, via PyVista sphere widgets)
  - Paper-validated metrics with equation references
  - Training curves (Fig 4a), q/force distributions
  - VAE diversity visualization
  - Export: CSV, DXF, OBJ, STL, JSON, PNG

Usage:
    python interactive_designer.py
"""
import os
import sys
import time

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

os.environ["QT_API"] = "pyside6"
import jax
import jax.numpy as jnp
import jax.random as jrn
import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from neural_fdm import DATA
from neural_fdm.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_mesh_from_generator,
    build_neural_model,
)
from neural_fdm.generators.grids import calculate_grid_from_tile_quarter
from neural_fdm.helpers import (
    edges_forces,
    edges_lengths,
    edges_vectors,
    vertices_residuals_from_xyz,
)
from neural_fdm.serialization import load_model

SEED = 90

TASKS = {
    "Bezier Shell": {"name": "bezier", "seed": 90},
    "Tower": {"name": "tower", "seed": 90},
}

COLOR_MODES = [
    ("Force Density q", "coolwarm_r"),
    ("Axial Force F=q*L", "coolwarm_r"),
    ("Shape Error per node", "turbo"),
]


def load_models(task_name="bezier"):
    """Load deterministic formfinder + optional VAE model for a given task."""
    cfg_path = os.path.join(os.path.dirname(__file__), f"{task_name}.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    key = jrn.PRNGKey(SEED)
    mk, _ = jax.random.split(key, 2)
    gen = build_data_generator(cfg)
    st = build_connectivity_structure_from_generator(cfg, gen)
    sk = build_neural_model("formfinder", cfg, gen, mk)
    mdl = load_model(os.path.join(DATA, f"formfinder_{task_name}.eqx"), sk)

    # Determine input size from generator
    test_xyz = gen(jrn.PRNGKey(0))
    input_size = test_xyz.shape[0]

    @jax.jit
    def pred(x):
        xh, (q, xf, ld) = mdl(x, st, aux_data=True)
        return xh, q, ld
    pred(jnp.zeros(input_size))

    # Try loading VAE model
    vae_model = None
    vae_path = os.path.join(DATA, f"variational_formfinder_variational_{task_name}.eqx")
    if os.path.exists(vae_path):
        try:
            vae_sk = build_neural_model("variational_formfinder", cfg, gen, mk)
            vae_model = load_model(vae_path, vae_sk)
            print(f"VAE model loaded for {task_name}.")
        except Exception as e:
            print(f"VAE model not loaded: {e}")

    return mdl, st, gen, pred, cfg, vae_model


def get_edges(nu):
    e = []
    for i in range(nu):
        for j in range(nu):
            idx = i * nu + j
            if j < nu - 1: e.append([idx, idx + 1])
            if i < nu - 1: e.append([idx, idx + nu])
            if i < nu - 1 and j < nu - 1: e.append([idx, idx + nu + 1])
    return np.array(e)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, w=3.8, h=2.5, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(w, h), dpi=dpi)
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class LSlider(QWidget):
    """Compact labeled slider."""
    def __init__(self, label, mn, mx, default, dec=2, cb=None):
        super().__init__()
        self.d, self.cb, self.s = dec, cb, 10**dec
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        l = QLabel(label); l.setFixedWidth(85); l.setFont(QFont("Segoe UI", 8))
        lay.addWidget(l)
        from PySide6.QtWidgets import QSlider
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(int(mn*self.s)); self.sl.setMaximum(int(mx*self.s))
        self.sl.setValue(int(default*self.s)); self.sl.valueChanged.connect(self._c)
        lay.addWidget(self.sl)
        self.vl = QLabel(f"{default:.{dec}f}")
        self.vl.setFixedWidth(42); self.vl.setAlignment(Qt.AlignRight)
        self.vl.setFont(QFont("Consolas", 8)); lay.addWidget(self.vl)
    def _c(self, v):
        val = v/self.s; self.vl.setText(f"{val:.{self.d}f}")
        if self.cb: self.cb(val)
    def value(self): return self.sl.value()/self.s


class MainWin(QMainWindow):
    def _make_slider(self, label, mn, mx, default, layout):
        s = LSlider(label, mn, mx, default, 2, self._up)
        layout.addWidget(s)
        return s

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAE-FDM Explorer")
        self.resize(1900, 1050)

        # App icon
        from PySide6.QtGui import QIcon
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.current_task = "bezier"
        self.tower_loss = 0.0
        print("Loading models...")
        self._load_task("bezier")
        self.color_idx = 0
        self._sample_actors = []
        self._sample_xyz_list = []
        self._sample_idx = 0
        self._sample_timer = QTimer(self)
        self._sample_timer.setInterval(300)
        self._sample_timer.timeout.connect(self._tick_sample)
        self._init_ui()

    def _load_task(self, task_name):
        """Load model, generator, structure, mesh, and default transform for a task."""
        self.current_task = task_name
        self.mdl, self.st, self.gen, self.pred_fn, self.cfg, self.vae = load_models(task_name)
        mesh = build_mesh_from_generator(self.cfg, self.gen)
        self.edges = np.array(list(mesh.edges()))

        if task_name == "bezier":
            self.transform = np.array([[0, 0, 3.0], [0, 0, 1.5], [0, 0, 0], [0, 0, 0]])
        else:
            # Tower: (radii, angles) for 3 rings
            # Default: all rings at (0.75, 0.75) radius scale, 0° rotation
            radii = np.array([[0.75, 0.75], [0.75, 0.75], [0.75, 0.75]])
            angles = np.array([0.0, 0.0, 0.0])
            self.transform = (jnp.array(radii), jnp.array(angles))

    def _build_task_sliders(self):
        """Build sliders appropriate for the current task."""
        # Clear existing sliders
        while self.ctrl_layout.count():
            item = self.ctrl_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        from PySide6.QtWidgets import QCheckBox

        self._spheres_unlocked = False
        self._sphere_widgets_active = False

        if self.current_task == "bezier":
            self.chk_unlock = QCheckBox("Unlock 3D sphere dragging")
            self.chk_unlock.setFont(QFont("Segoe UI", 8))
            self.chk_unlock.setChecked(False)
            self.chk_unlock.stateChanged.connect(self._on_unlock_toggle)
            self.ctrl_layout.addWidget(self.chk_unlock)
            self.sl_h = self._make_slider("c1.z height", 1.0, 10.0, 3.0, self.ctrl_layout)
            self.sl_sx = self._make_slider("c2.x spread", -5.0, 5.0, 0.0, self.ctrl_layout)
            self.sl_ez = self._make_slider("c2.z edge", 0.0, 10.0, 1.5, self.ctrl_layout)
            self.sl_cy = self._make_slider("c3.y curve", -5.0, 5.0, 0.0, self.ctrl_layout)
        else:
            # Tower: 9 params (3 rings × r1, r2, angle)
            # Paper trains with ALL rings varying independently
            ring_names = ["Bottom ring", "Middle ring", "Top ring"]
            self.tower_sliders = {}
            for i, name in enumerate(ring_names):
                lbl = QLabel(name)
                lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
                self.ctrl_layout.addWidget(lbl)
                self.tower_sliders[f"r1_{i}"] = self._make_slider("  Radius 1", 0.5, 1.5, 0.75, self.ctrl_layout)
                self.tower_sliders[f"r2_{i}"] = self._make_slider("  Radius 2", 0.5, 1.5, 0.75, self.ctrl_layout)
                self.tower_sliders[f"rot_{i}"] = self._make_slider("  Rotation °", -15.0, 15.0, 0.0, self.ctrl_layout)

    def _on_task_change(self, label):
        """Handle task selector change."""
        task_info = TASKS.get(label)
        if not task_info or task_info["name"] == self.current_task:
            return
        print(f"Switching to {label}...")
        self._load_task(task_info["name"])
        if hasattr(self, '_tower_loss_fn'):
            del self._tower_loss_fn
        self._build_task_sliders()
        if self.vae_group is not None:
            self.vae_group.setVisible(self.vae is not None)
        self._meshes_created = False
        self.pl.clear()
        self.pl.clear_sphere_widgets()
        self._compute_and_render()
        self.pl.reset_camera()

    def _init_ui(self):
        """Build the UI layout (called once from __init__)."""
        c = QWidget(); self.setCentralWidget(c)
        ml = QHBoxLayout(c); ml.setContentsMargins(4,4,4,4); ml.setSpacing(4)

        # LEFT
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFixedWidth(280)
        sc.setWidget(self._build_ctrl()); ml.addWidget(sc)

        # CENTER
        self.pl = QtInteractor(self)
        self.pl.set_background("white", top="aliceblue")
        ml.addWidget(self.pl, stretch=3)

        # RIGHT
        self.tabs = QTabWidget(); self.tabs.setFixedWidth(430)
        self._build_tabs(); ml.addWidget(self.tabs)

        self._compute_and_render()

    # =====================================================================
    # LEFT PANEL
    # =====================================================================
    def _build_ctrl(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)

        # Task selector
        g = QGroupBox("Task")
        gl = QVBoxLayout(g)
        self.task_combo = QComboBox()
        for label in TASKS:
            self.task_combo.addItem(label)
        self.task_combo.currentTextChanged.connect(self._on_task_change)
        gl.addWidget(self.task_combo)
        lay.addWidget(g)

        # Control points
        self.ctrl_group = QGroupBox("Control Points")
        self.ctrl_layout = QVBoxLayout(self.ctrl_group)
        self._build_task_sliders()
        lay.addWidget(self.ctrl_group)

        # Color mode
        g = QGroupBox("Edge Coloring")
        gl = QVBoxLayout(g)
        self.cc = QComboBox()
        for nm, _ in COLOR_MODES: self.cc.addItem(nm)
        self.cc.currentIndexChanged.connect(self._on_color)
        gl.addWidget(self.cc)
        lay.addWidget(g)

        # VAE diversity (shown/hidden based on task having VAE model)
        self.vae_group = None
        if True:  # Always create, visibility controlled by task
            g = QGroupBox("Solution Diversity (VAE)")
            self.vae_group = g
            gl = QVBoxLayout(g)
            btn = QPushButton("Sample diverse equilibria")
            btn.clicked.connect(self._generate_diversity)
            gl.addWidget(btn)
            self.vae_label = QLabel(
                "Explore the space of valid force density\n"
                "solutions for the current target shape.\n"
                "Each sample is in equilibrium.")
            self.vae_label.setFont(QFont("Segoe UI", 8))
            self.vae_label.setWordWrap(True)
            gl.addWidget(self.vae_label)
            lay.addWidget(g)
            g.setVisible(self.vae is not None)

        # Export
        g = QGroupBox("Export")
        gl = QVBoxLayout(g)
        for txt, fn in [
            ("CSV  - Geometry + q", self._exp_csv),
            ("DXF  - Centerlines", self._exp_dxf),
            ("OBJ  - Mesh", self._exp_obj),
            ("STL  - 3D print", self._exp_stl),
            ("JSON - Full data", self._exp_json),
            ("PNG  - Screenshot", self._exp_png),
        ]:
            b = QPushButton(txt); b.clicked.connect(fn); gl.addWidget(b)
        lay.addWidget(g)

        lay.addStretch()
        leg = QLabel(
            "VAE-FDM\n"
            "github.com/efradeca/vae-fdm\n"
            "Efrain Deulofeu, 2026\n\n"
            "Red = control points\n"
            "Gray wireframe = target\n"
            "Blue surface = prediction\n"
            "Edges = force density\n\n"
            "Citing:\n"
            "Pastrana, R., Medina, E., de Oliveira,\n"
            "I.M., Adriaenssens, S., Adams, R.P.\n"
            "ICLR 2025. arXiv:2409.02606\n\n"
            "Research and educational use only.\n"
            "Not a structural design tool.")
        leg.setFont(QFont("Segoe UI", 7))
        leg.setStyleSheet("color:#777; padding:4px;")
        leg.setWordWrap(True); lay.addWidget(leg)
        return w

    # =====================================================================
    # RIGHT TABS
    # =====================================================================
    def _build_tabs(self):
        # Metrics
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas", 9))
        self.txt.setStyleSheet("background:#fafafa; padding:6px;")
        self.tabs.addTab(self.txt, "Metrics")

        # q histogram
        self.canvas_q = MplCanvas(); self.tabs.addTab(self.canvas_q, "q Dist")

        # Force histogram
        self.canvas_f = MplCanvas(); self.tabs.addTab(self.canvas_f, "Forces")

        # Shape error
        self.canvas_err = MplCanvas(); self.tabs.addTab(self.canvas_err, "Shape Err")

        # Training curves (Fig 4a) - load from saved data
        self.canvas_train = MplCanvas(); self.tabs.addTab(self.canvas_train, "Training")
        self._plot_training_curves()

        # VAE diversity tab (if model available)
        if self.vae is not None:
            self.canvas_div = MplCanvas(); self.tabs.addTab(self.canvas_div, "VAE Diversity")

    def _plot_training_curves(self):
        """Plot training curves from saved loss files (task-specific)."""
        ax = self.canvas_train.ax; ax.clear()
        try:
            task = self.current_task
            shape_f = os.path.join(DATA, f"losses_formfinder_{task}_shape_error.txt")
            loss_f = os.path.join(DATA, f"losses_formfinder_{task}_loss.txt")
            if os.path.exists(loss_f):
                loss = np.loadtxt(loss_f)
                ax.semilogy(loss, label="Loss", color="#1565c0", alpha=0.7)
                if os.path.exists(shape_f):
                    shape = np.loadtxt(shape_f)
                    ax.semilogy(shape, label="L_shape", color="#2e7d32", alpha=0.7)
                ax.set_xlabel("Training Step"); ax.set_ylabel("Loss (log)")
                ax.set_title(f"Training Curves ({task})", fontsize=9)
                ax.legend(fontsize=7)
            elif os.path.exists(shape_f):
                shape = np.loadtxt(shape_f)
                ax.semilogy(shape, label="L_shape", color="#1565c0", alpha=0.7)
                ax.set_xlabel("Training Step"); ax.set_ylabel("Loss (log)")
                ax.set_title(f"Training Curves ({task})", fontsize=9)
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, f"Train model first:\npython train.py formfinder {task}",
                        ha='center', va='center', fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, "Loss data not found", ha='center', va='center')
        ax.tick_params(labelsize=7)
        self.canvas_train.draw()

    # =====================================================================
    # COMPUTATION
    # =====================================================================
    def _compute(self):
        t0 = time.perf_counter()
        if self.current_task == "bezier":
            self.transform[0, 2] = self.sl_h.value()
            self.transform[1, 0] = self.sl_sx.value()
            self.transform[1, 2] = self.sl_ez.value()
            self.transform[2, 1] = self.sl_cy.value()
            xyz_t = self.gen.evaluate_points(jnp.array(self.transform))
        else:
            # Tower: 9 params (3 rings × r1, r2, angle)
            radii = []
            angles = []
            for i in range(3):
                r1 = self.tower_sliders[f"r1_{i}"].value()
                r2 = self.tower_sliders[f"r2_{i}"].value()
                rot = self.tower_sliders[f"rot_{i}"].value()
                radii.append([r1, r2])
                angles.append(rot)
            self.transform = (jnp.array(radii), jnp.array(angles))
            xyz_t = self.gen.evaluate_points(self.transform)
        self.tnp = np.array(xyz_t).reshape(-1, 3)
        pred, q, ld = self.pred_fn(xyz_t)
        self.pnp = np.array(pred).reshape(-1, 3)
        self.q = np.array(q); self.ld = np.array(ld)

        xj = jnp.reshape(pred, (-1, 3))
        v = edges_vectors(xj, self.st.connectivity)
        l = edges_lengths(v)
        f = edges_forces(jnp.array(self.q), l)
        self.F = np.array(f).flatten(); self.L = np.array(l).flatten()

        res = vertices_residuals_from_xyz(jnp.array(self.q), jnp.array(self.ld), xj, self.st)
        self.res_np = np.array(res)
        self.max_res = float(np.max(np.abs(self.res_np)))
        self.res_mag = np.linalg.norm(self.res_np, axis=1)
        if self.current_task == "tower":
            # Tower shape error: L2 on compression ring vertices only
            # (matches paper metric, avoids expensive second forward pass)
            shape_tube_t = self.tnp.reshape(self.gen.shape_tube)
            shape_tube_p = self.pnp.reshape(self.gen.shape_tube)
            rings_idx = self.gen.levels_rings_comp
            target_rings = shape_tube_t[rings_idx]
            pred_rings = shape_tube_p[rings_idx]
            self.err_l1 = float(np.sqrt(np.sum((target_rings - pred_rings) ** 2)))
            self.tower_loss = self.err_l1
        else:
            self.err_l1 = float(np.sum(np.abs(self.tnp - self.pnp)))
        self.err_node = np.linalg.norm(self.tnp - self.pnp, axis=1)
        self.all_comp = bool(np.all(self.q <= 0.001))
        self.dt = (time.perf_counter() - t0) * 1000

    def _get_sc(self):
        ne = len(self.edges)
        if self.color_idx == 0: r = self.q
        elif self.color_idx == 1: r = self.F
        elif self.color_idx == 2:
            r = np.array([(self.err_node[e[0]]+self.err_node[e[1]])/2 for e in self.edges])
        else: r = self.q
        return r[:ne] if len(r) >= ne else np.concatenate([r, np.zeros(ne-len(r))])

    def _compute_and_render(self):
        if getattr(self, '_sample_timer', None) and self._sample_timer.isActive():
            self._sample_timer.stop()
        sa = getattr(self, '_sample_actors', None)
        if sa:
            for actor in sa:
                try:
                    self.pl.remove_actor(actor)
                except Exception:
                    pass
            self._sample_actors = []
        self._compute()
        if not getattr(self, '_meshes_created', False):
            self._create_3d()
            self._meshes_created = True
        else:
            self._update_3d()
        self._update_txt()
        self._update_charts()

    # =====================================================================
    # 3D: Create once, update in-place (no flicker)
    # =====================================================================
    def _create_3d(self):
        """Initial 3D setup - called once."""
        self._target_mesh = None
        self._pred_mesh = None
        self._cp_all_poly = None
        self._cp_unique_poly = None

        if self.current_task == "bezier":
            nu = self.cfg["generator"]["num_uv"]
            # Target wireframe
            xt = self.tnp.reshape(nu, nu, 3)
            self._target_mesh = pv.StructuredGrid(xt[:, :, 0], xt[:, :, 1], xt[:, :, 2])
            self.pl.add_mesh(self._target_mesh, color="gray", style="wireframe",
                             line_width=1, opacity=0.3)

            # Predicted surface
            xp = self.pnp.reshape(nu, nu, 3)
            self._pred_mesh = pv.StructuredGrid(xp[:, :, 0], xp[:, :, 1], xp[:, :, 2])
            self.pl.add_mesh(self._pred_mesh, color="steelblue", opacity=0.4,
                             show_edges=False)
        else:
            # Tower: show target as wireframe edges (same connectivity as prediction)
            target_lines = []
            for e in self.edges:
                target_lines.extend([2, e[0], e[1]])
            self._target_poly = pv.PolyData(self.tnp, lines=np.array(target_lines))
            self.pl.add_mesh(self._target_poly, color="gray", line_width=1, opacity=0.3)

        # Structural edges (same for both tasks)
        sc = self._get_sc()
        _, cm = COLOR_MODES[self.color_idx]
        lines = []
        for e in self.edges:
            lines.extend([2, e[0], e[1]])
        self._edge_poly = pv.PolyData(self.pnp, lines=np.array(lines))
        lbl = COLOR_MODES[self.color_idx][0]
        self._edge_poly[lbl] = sc
        mx = abs(sc).max() if len(sc) > 0 else 1.0
        fmt = "%.2e" if (0 < mx < 0.01) else "%.3f"
        self._edge_actor = self.pl.add_mesh(
            self._edge_poly, scalars=lbl, cmap=cm, line_width=3,
            scalar_bar_args={"title": lbl, "n_labels": 4, "fmt": fmt})

        # Task-specific overlays
        if self.current_task == "bezier":
            # Control points: mirrored (visual only)
            tile = np.array(self.gen.surface.grid.tile)
            cp_unique = tile + self.transform
            cp_all = np.array(calculate_grid_from_tile_quarter(jnp.array(cp_unique)))
            self._cp_all_poly = pv.PolyData(cp_all)
            self.pl.add_points(self._cp_all_poly, color="orange", point_size=6,
                               render_points_as_spheres=True, opacity=0.4)

            self._tile = tile
            if getattr(self, '_sphere_widgets_active', False):
                for i in range(4):
                    self.pl.add_sphere_widget(
                        self._make_sphere_cb(i),
                        center=cp_unique[i].tolist(),
                        radius=0.15,
                        color="red",
                        style="surface",
                        interaction_event="end",
                    )
            else:
                self._cp_unique_poly = pv.PolyData(cp_unique)
                self.pl.add_points(self._cp_unique_poly, color="red", point_size=14,
                                   render_points_as_spheres=True)
        else:
            # Tower: draw compression ring polygons (like paper Figure 9)
            try:
                shape_tube = self.gen.shape_tube
                rings_idx = self.gen.levels_rings_comp
                xyz_tube = self.pnp.reshape(shape_tube)
                for ri in rings_idx:
                    ring_pts = xyz_tube[ri]
                    # Close the ring
                    ring_closed = np.vstack([ring_pts, ring_pts[0:1]])
                    ring_poly = pv.PolyData(ring_closed)
                    ring_poly.lines = np.array(
                        [ring_closed.shape[0]] + list(range(ring_closed.shape[0]))
                    )
                    self.pl.add_mesh(ring_poly, color="darkgray", line_width=4, opacity=0.7)
            except Exception:
                pass  # graceful fallback if ring indices don't match

        self.pl.add_axes()
        self.pl.reset_camera()

    def _on_unlock_toggle(self, state):
        """Toggle between static points and draggable sphere widgets."""
        self._spheres_unlocked = bool(state)
        if state:
            QMessageBox.warning(self, "3D Control Point Editing",
                "The neural model was trained on doubly-symmetric Bezier\n"
                "surfaces with specific parameter ranges (paper Table 6,\n"
                "Pastrana et al. ICLR 2025).\n\n"
                "Dragging control points is constrained to the trained axes:\n"
                "  c1: Z axis only (shell height)\n"
                "  c2: X and Z axes (horizontal spread and edge height)\n"
                "  c3: Y axis only (lateral curvature)\n"
                "  c4: fixed (corner anchor, not movable)\n\n"
                "Moving points beyond trained ranges may reduce prediction\n"
                "accuracy. The equilibrium guarantee (R=0) is maintained\n"
                "by the FDM decoder regardless of input quality.")
        # Full re-render to switch between static points and sphere widgets
        self._sphere_widgets_active = bool(state)
        self._meshes_created = False
        self.pl.clear_sphere_widgets()  # Remove VTK sphere widgets explicitly
        self.pl.clear()
        self._create_3d_with_mode()

    def _make_sphere_cb(self, idx):
        """Create callback for draggable sphere widget idx.

        Constrains movement to paper Table 6 axes:
          c1 (idx=0): only Z
          c2 (idx=1): X and Z
          c3 (idx=2): only Y
          c4 (idx=3): fixed
        """
        # Allowed axes per control point (paper Table 6)
        allowed = {0: [2], 1: [0, 2], 2: [1], 3: []}

        def cb(point):
            # Only called when sphere widgets exist (unlocked mode)
            new_pos = np.array(point)
            new_t = new_pos - self._tile[idx]

            # Constrain to allowed axes only
            for d in range(3):
                if d in allowed.get(idx, []):
                    self.transform[idx, d] = new_t[d]

            # Sync sliders (no callback trigger)
            self._syncing = True
            self.sl_h.sl.setValue(int(self.transform[0, 2] * self.sl_h.s))
            self.sl_sx.sl.setValue(int(self.transform[1, 0] * self.sl_sx.s))
            self.sl_ez.sl.setValue(int(self.transform[1, 2] * self.sl_ez.s))
            self.sl_cy.sl.setValue(int(self.transform[2, 1] * self.sl_cy.s))
            self._syncing = False

            # Update in-place (no clear, no flicker)
            self._compute()
            self._update_3d()
            self._update_txt()
        return cb

    def _update_3d(self):
        """Update mesh points in-place (no flicker)."""
        if self.current_task == "bezier":
            nu = self.cfg["generator"]["num_uv"]
            # Update target wireframe
            if self._target_mesh is not None:
                xt = self.tnp.reshape(nu, nu, 3)
                new_target = pv.StructuredGrid(xt[:, :, 0], xt[:, :, 1], xt[:, :, 2])
                self._target_mesh.points = new_target.points

            # Update predicted surface
            if self._pred_mesh is not None:
                xp = self.pnp.reshape(nu, nu, 3)
                new_pred = pv.StructuredGrid(xp[:, :, 0], xp[:, :, 1], xp[:, :, 2])
                self._pred_mesh.points = new_pred.points
        else:
            # Tower: update target wireframe
            if hasattr(self, '_target_poly'):
                self._target_poly.points = self.tnp.copy()

        # Update edge positions, scalars, and scalar bar range (same for both)
        self._edge_poly.points = self.pnp.copy()
        sc = self._get_sc()
        lbl = COLOR_MODES[self.color_idx][0]
        self._edge_poly[lbl] = sc
        if hasattr(self, '_edge_actor') and self._edge_actor is not None:
            mapper = self._edge_actor.GetMapper()
            if mapper is not None:
                vmin, vmax = float(sc.min()), float(sc.max())
                if abs(vmax - vmin) < 1e-10:
                    vmax = vmin + 1e-6
                mapper.SetScalarRange(vmin, vmax)

        # Update control points (bezier only)
        if self.current_task == "bezier":
            tile = np.array(self.gen.surface.grid.tile)
            cp_unique = tile + self.transform
            cp_all = np.array(calculate_grid_from_tile_quarter(jnp.array(cp_unique)))
            if self._cp_all_poly is not None:
                self._cp_all_poly.points = cp_all
            if hasattr(self, '_cp_unique_poly') and self._cp_unique_poly is not None and not self._sphere_widgets_active:
                self._cp_unique_poly.points = cp_unique

        self.pl.render()

    def _create_3d_with_mode(self):
        """Re-create 3D with current sphere mode."""
        self._create_3d()
        self._meshes_created = True

    def _render_3d(self):
        """Full re-render (used for color mode change)."""
        self._meshes_created = False
        self.pl.clear()
        self._create_3d()


    # =====================================================================
    # TEXT + CHARTS
    # =====================================================================
    def _update_txt(self):
        if self.current_task == "bezier":
            cc = "#2e7d32" if self.all_comp else "#c62828"
            cs = "Yes (all q&lt;=0)" if self.all_comp else "NO"
        else:
            # Tower has mixed tension/compression by design
            n_comp = int(np.sum(self.q < -0.001))
            n_tens = int(np.sum(self.q > 0.001))
            cc = "#1565c0"
            cs = f"Mixed ({n_comp} comp, {n_tens} tens)"
        vae_str = ""
        if self.vae:
            vae_str = """
<h3 style="color:#555;">Solution Diversity</h3>
<p style="font-size:8pt;">VAE encoder enables sampling multiple valid
equilibria. Use the panel to explore the solution space
(cf. Veenendaal &amp; Block 2012 on non-uniqueness).</p>"""

        html = f"""
<h3 style="color:#1565c0;">Equilibrium (Eq. 1)</h3>
<table style="font-family:Consolas; font-size:9pt;">
<tr><td>max |R|:</td><td><b>{'%.2e' % self.max_res}</b></td></tr>
<tr><td>Compression:</td><td style="color:{cc}"><b>{cs}</b></td></tr>
</table>
<h3 style="color:#1565c0;">Force Densities q (Eq. 8)</h3>
<table style="font-family:Consolas; font-size:9pt;">
<tr><td>q:</td><td>[{self.q.min():.4f}, {self.q.max():.4f}]</td></tr>
<tr><td>mean:</td><td>{self.q.mean():.4f} (std: {self.q.std():.4f})</td></tr>
</table>
<h3 style="color:#1565c0;">Forces F = q*L</h3>
<table style="font-family:Consolas; font-size:9pt;">
<tr><td>F:</td><td>[{self.F.min():.4f}, {self.F.max():.4f}]</td></tr>
<tr><td>|F| mean:</td><td>{np.abs(self.F).mean():.4f}</td></tr>
</table>
<h3 style="color:#1565c0;">Shape Match</h3>
<table style="font-family:Consolas; font-size:9pt;">
<tr><td>{'L_shape (L1):' if self.current_task == 'bezier' else 'Shape err (L2 rings):'}</td><td><b>{self.err_l1:.2f}</b>{' (paper: 3.0+/-2.0)' if self.current_task == 'bezier' else ' (paper: 1.4+/-0.4)'}</td></tr>
{'<tr><td>Total loss:</td><td><b>' + f'{self.tower_loss:.2f}' + '</b></td></tr>' if self.current_task == 'tower' else ''}
<tr><td>Time:</td><td><b>{self.dt:.1f} ms</b></td></tr>
</table>
{vae_str}
<hr><p style="font-size:7pt; color:#aaa;">arXiv:2409.02606</p>"""
        self.txt.setHtml(html)

    def _update_charts(self):
        ax = self.canvas_q.ax; ax.clear()
        ax.hist(self.q, bins=30, color="#1565c0", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, label="q=0")
        ax.set_xlabel("q"); ax.set_ylabel("Count")
        ax.set_title("Force Density Distribution", fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        self.canvas_q.draw()

        ax = self.canvas_f.ax; ax.clear()
        ax.hist(self.F, bins=30, color="#e65100", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("F = q*L"); ax.set_ylabel("Count")
        ax.set_title("Axial Force Distribution", fontsize=9)
        ax.tick_params(labelsize=7)
        self.canvas_f.draw()

        ax = self.canvas_err.ax; ax.clear()
        ax.barh(range(len(self.err_node)), np.sort(self.err_node)[::-1],
                color="#2e7d32", alpha=0.8)
        ax.axvline(x=self.err_node.mean(), color="red", linestyle="--",
                    label=f"Mean: {self.err_node.mean():.3f}")
        ax.set_ylabel("Node"); ax.set_xlabel("|X(q) - X_hat|")
        ax.set_title("Shape Error per Node (Eq. 5)", fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        self.canvas_err.draw()

    # =====================================================================
    # VAE DIVERSITY
    # =====================================================================
    def _generate_diversity(self):
        """Sample diverse equilibrium solutions and animate them in the 3D viewport."""
        if self.vae is None:
            return
        from neural_fdm.variational import compute_diversity_metrics, compute_variance_per_edge

        if self._sample_timer.isActive():
            self._sample_timer.stop()
        for actor in self._sample_actors:
            try:
                self.pl.remove_actor(actor)
            except Exception:
                pass
        self._sample_actors = []

        if self.current_task == "bezier":
            xyz_t = self.gen.evaluate_points(jnp.array(self.transform))
        else:
            xyz_t = self.gen.evaluate_points(self.transform)
        key = jrn.PRNGKey(int(time.time()) % 10000)

        x_hats, qs = self.vae.sample(xyz_t, self.st, key, num_samples=40)
        metrics = compute_diversity_metrics(x_hats, qs)
        compute_variance_per_edge(self.vae, xyz_t, self.st, key, n_samples=20)

        x_hats_np = np.array(x_hats)
        xyz_t_np = np.array(xyz_t)
        diff = x_hats_np.reshape(x_hats_np.shape[0], -1) - xyz_t_np.reshape(-1)[None]
        errors = np.linalg.norm(diff, axis=1)

        order = list(range(len(x_hats_np)))
        rng = np.random.default_rng(int(time.time()) % 10000)
        rng.shuffle(order)

        det_xyz = np.array(self.pnp)
        det_err = float(np.linalg.norm(det_xyz.reshape(-1) - xyz_t_np.reshape(-1)))

        qs_np = np.array(qs)
        det_q = np.array(self.q).reshape(-1)
        if det_q.shape[0] != qs_np.shape[1]:
            det_q = det_q[: qs_np.shape[1]] if det_q.shape[0] > qs_np.shape[1] else np.pad(
                det_q, (0, qs_np.shape[1] - det_q.shape[0]))

        self._sample_xyz_list = [x_hats_np[i] for i in order] + [det_xyz]
        self._sample_errors = [float(errors[i]) for i in order] + [det_err]
        self._qs_frames = [qs_np[i] for i in order] + [det_q]
        self._sample_total = len(self._sample_xyz_list)

        q_std = np.array(metrics["q_std_per_edge"])
        self._sort_idx = np.argsort(q_std)[::-1]
        n_edges = len(self._sort_idx)

        fig = self.canvas_div.figure
        fig.patch.set_facecolor("white")
        ax = self.canvas_div.ax
        ax.clear()
        if getattr(self, '_canvas_div_ax2', None) is not None:
            try:
                self._canvas_div_ax2.remove()
            except Exception:
                pass
            self._canvas_div_ax2 = None

        ax.set_facecolor("#fcfcfc")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#2b2b2b")
            ax.spines[side].set_linewidth(1.1)
        ax.tick_params(direction="in", length=4.5, width=0.9,
                       colors="#222", labelsize=9)
        ax.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.5,
                color="#b4bac3", alpha=0.55, zorder=0)

        ax.bar(np.arange(n_edges), q_std[self._sort_idx],
               width=1.0, color="#f5b041", alpha=0.42,
               edgecolor="none", zorder=1, label="Design freedom (population)")
        ax.set_ylabel(r"$\sigma_q$  (population std)",
                      fontsize=10, color="#7a4f00", labelpad=6)
        ax.tick_params(axis="y", labelsize=9, colors="#7a4f00")

        ax2 = ax.twinx()
        self._canvas_div_ax2 = ax2
        for side in ("top",):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color("#2b2b2b")
        ax2.spines["right"].set_linewidth(1.1)
        ax2.tick_params(direction="in", length=4.5, width=0.9,
                        colors="#222", labelsize=9)

        all_q = np.concatenate(self._qs_frames)
        q_lo, q_hi = float(all_q.min()), float(all_q.max())
        pad = 0.1 * max(abs(q_lo), abs(q_hi), 1e-6)
        ax2.set_ylim(q_lo - pad, q_hi + pad)
        ax2.set_ylabel(r"Force density  $q_i$",
                       fontsize=10, color="#222", labelpad=6)

        x_edges = np.arange(n_edges)
        self._ghost_lines = []
        for _ in range(self._sample_total - 1):
            ln, = ax2.plot([], [], color="#6b7280",
                           linewidth=0.9, alpha=0.0, zorder=3)
            self._ghost_lines.append(ln)

        self._current_line, = ax2.plot(
            [], [], color="#c62828", linewidth=2.4, alpha=0.95,
            zorder=6, label="Current iteration")
        self._det_line, = ax2.plot(
            [], [], color="#0d47a1", linewidth=3.0, alpha=0.0,
            zorder=7, label="Deterministic FDM")

        ax.set_xlim(-0.5, n_edges - 0.5)
        ax.set_xlabel("Edge  (sorted by design freedom)",
                      fontsize=9, color="#222", labelpad=5)
        ax.set_title(
            f"VAE ensemble  ·  {self._sample_total - 1} samples + deterministic",
            fontsize=10, fontweight="bold", color="#1a1a1a", pad=8)

        h_bar, l_bar = ax.get_legend_handles_labels()
        h_lin, l_lin = ax2.get_legend_handles_labels()
        leg = ax2.legend(
            h_bar + h_lin, l_bar + l_lin,
            loc="upper right", fontsize=9, frameon=True,
            fancybox=False, edgecolor="#555", framealpha=0.96,
            borderpad=0.6, handletextpad=0.7, labelspacing=0.4)
        leg.get_frame().set_linewidth(0.8)

        try:
            fig.tight_layout()
        except Exception:
            pass
        self.canvas_div.draw()

        self._diversity_metrics = metrics
        self._sample_idx = 0
        self._tick_sample()
        self._sample_timer.start()

    def _refresh_sample_label(self, is_final=False):
        m = getattr(self, '_diversity_metrics', None)
        if m is None:
            return
        total = getattr(self, '_sample_total', len(self._sample_xyz_list))
        idx_display = min(self._sample_idx, total)
        if is_final:
            err = self._sample_errors[-1]
            self.vae_label.setText(
                f"Converged  —  deterministic FDM\n"
                f"  Shape error (L2): {err:.3f}\n"
                f"  Shape diversity (L1): {m['shape_pairwise_L1_mean']:.2f}\n"
                f"  q std across samples: {m['q_std_mean']:.4f}\n\n"
                f"VAE explored the solution space;\n"
                f"final surface is Pastrana et al.'s\n"
                f"deterministic FDM prediction."
            )
        else:
            self.vae_label.setText(
                f"Exploring {idx_display}/{total - 1}  —  VAE samples\n"
                f"  Shape diversity (L1): {m['shape_pairwise_L1_mean']:.2f}\n"
                f"  q std across samples: {m['q_std_mean']:.4f}\n"
                f"  All satisfy equilibrium.\n\n"
                f"Morphing through diverse force\n"
                f"density solutions for the target."
            )

    def _tick_sample(self):
        import matplotlib.cm as cm
        if not self._sample_xyz_list:
            return
        for actor in self._sample_actors:
            try:
                self.pl.remove_actor(actor)
            except Exception:
                pass
        self._sample_actors = []

        total = self._sample_total
        cur = self._sample_idx
        is_final_frame = (cur == total - 1)
        xyz = self._sample_xyz_list[cur]

        if is_final_frame:
            color = (0.082, 0.396, 0.753)
            opacity = 0.9
        else:
            frac = cur / max(total - 2, 1)
            rgba = cm.turbo(frac)
            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            opacity = 0.6

        if self.current_task == "bezier":
            nu = self.cfg["generator"]["num_uv"]
            xp = xyz.reshape(nu, nu, 3)
            sg = pv.StructuredGrid(xp[:, :, 0], xp[:, :, 1], xp[:, :, 2])
            actor = self.pl.add_mesh(sg, color=color, opacity=opacity, show_edges=False)
            self._sample_actors.append(actor)
        else:
            lines = []
            for e in self.edges:
                lines.extend([2, e[0], e[1]])
            pd = pv.PolyData(xyz, lines=np.array(lines))
            actor = self.pl.add_mesh(pd, color=color, line_width=3, opacity=min(opacity + 0.25, 1.0))
            self._sample_actors.append(actor)
        self.pl.render()

        cur_line = getattr(self, '_current_line', None)
        if cur_line is not None and getattr(self, '_qs_frames', None):
            q_cur = self._qs_frames[cur]
            q_ord = q_cur[self._sort_idx]
            x_edges = np.arange(len(q_ord))

            if not is_final_frame:
                old_x, old_y = cur_line.get_data()
                if len(old_x) > 0 and cur > 0:
                    ghost_idx = cur - 1
                    if ghost_idx < len(self._ghost_lines):
                        gl = self._ghost_lines[ghost_idx]
                        gl.set_data(old_x, old_y)
                        gl.set_alpha(0.18)

                cur_line.set_data(x_edges, q_ord)
                cur_line.set_color("#c62828")
                cur_line.set_linewidth(2.4)
                cur_line.set_alpha(0.95)
                self._det_line.set_data([], [])
                self._det_line.set_alpha(0.0)
            else:
                old_x, old_y = cur_line.get_data()
                if len(old_x) > 0:
                    ghost_idx = cur - 1
                    if ghost_idx < len(self._ghost_lines):
                        gl = self._ghost_lines[ghost_idx]
                        gl.set_data(old_x, old_y)
                        gl.set_alpha(0.18)
                cur_line.set_data([], [])
                cur_line.set_alpha(0.0)
                self._det_line.set_data(x_edges, q_ord)
                self._det_line.set_alpha(1.0)

            self.canvas_div.draw_idle()

        self._sample_idx += 1
        self._refresh_sample_label(is_final=is_final_frame)
        if is_final_frame:
            self._sample_timer.stop()

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def _up(self, _=None):
        if getattr(self, '_syncing', False):
            return
        # Guard: ensure correct sliders exist for current task
        if self.current_task == "bezier" and not hasattr(self, 'sl_h'):
            return
        if self.current_task != "bezier" and not hasattr(self, 'tower_sliders'):
            return
        self._compute_and_render()

    def _on_color(self, i):
        self.color_idx = i
        self._meshes_created = False
        self.pl.clear()
        self._create_3d()

    # =====================================================================
    # EXPORTS
    # =====================================================================
    def _ask(self, t, f, d):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getSaveFileName(self, t, os.path.join(DATA, d), f)
        return p if p else None

    def _exp_csv(self):
        p = self._ask("Save CSV", "CSV (*.csv)", "results.csv")
        if not p: return
        with open(p, "w") as f:
            f.write("# Neural FDM - arXiv:2409.02606\n")
            f.write(f"# L_shape={self.err_l1:.4f}\n")
            f.write("# NODES\nid,xt,yt,zt,xp,yp,zp,err\n")
            for i in range(len(self.tnp)):
                t, pp = self.tnp[i], self.pnp[i]
                f.write(f"{i},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f},{pp[0]:.6f},{pp[1]:.6f},{pp[2]:.6f},{self.err_node[i]:.6f}\n")
            f.write("# EDGES\nid,ni,nj,q,L,F\n")
            for i in range(min(len(self.edges), len(self.q))):
                e = self.edges[i]
                f.write(f"{i},{e[0]},{e[1]},{self.q[i]:.6f},{self.L[i]:.6f},{self.F[i]:.6f}\n")
        QMessageBox.information(self, "Export", f"Saved: {p}")

    def _exp_dxf(self):
        p = self._ask("Save DXF", "DXF (*.dxf)", "structure.dxf")
        if not p: return
        from neural_fdm.export import export_dxf
        export_dxf(p, self.pnp, self.edges)
        QMessageBox.information(self, "Export", f"DXF: {p}\n\nFor AutoCAD, Robot, ETABS, SAP2000.")

    def _exp_obj(self):
        p = self._ask("Save OBJ", "OBJ (*.obj)", "shape.obj")
        if not p: return
        from neural_fdm.export import export_obj
        if self.current_task == "bezier":
            nu = self.cfg["generator"]["num_uv"]
            faces = [[i * nu + j, i * nu + j + 1, (i + 1) * nu + j + 1, (i + 1) * nu + j]
                     for i in range(nu - 1) for j in range(nu - 1)]
            export_obj(p, self.pnp, np.array(faces))
        else:
            # Tower: export edges as OBJ lines
            export_obj(p, self.pnp, np.array(self.edges))
        QMessageBox.information(self, "Export", f"OBJ: {p}")

    def _exp_stl(self):
        p = self._ask("Save STL", "STL (*.stl)", "shape.stl")
        if not p: return
        if self.current_task == "bezier":
            nu = self.cfg["generator"]["num_uv"]
            xp = self.pnp.reshape(nu, nu, 3)
            pv.StructuredGrid(xp[:, :, 0], xp[:, :, 1], xp[:, :, 2]).extract_surface().save(p)
        else:
            QMessageBox.information(self, "Export", "STL not available for tower task (tube geometry)")
            return
        QMessageBox.information(self, "Export", f"STL: {p}")

    def _exp_json(self):
        p = self._ask("Save JSON", "JSON (*.json)", "data.json")
        if not p: return
        import json
        if self.current_task == "bezier":
            control_points = np.asarray(self.transform).tolist()
        else:
            radii, angles = self.transform
            control_points = {
                "radii": np.asarray(radii).tolist(),
                "angles": np.asarray(angles).tolist(),
            }
        data = {
            "paper": "Pastrana et al., ICLR 2025, arXiv:2409.02606",
            "task": self.current_task,
            "L_shape": round(self.err_l1, 4),
            "inference_ms": round(self.dt, 1),
            "control_points": control_points,
            "nodes": {"target": self.tnp.tolist(), "predicted": self.pnp.tolist()},
            "edges": {"connectivity": self.edges.tolist(), "q": self.q.tolist(),
                      "length": self.L.tolist(), "force": self.F.tolist()},
        }
        with open(p, "w") as f: json.dump(data, f, indent=2)
        QMessageBox.information(self, "Export", f"JSON: {p}")

    def _exp_png(self):
        p = self._ask("Save PNG", "PNG (*.png)", "screenshot.png")
        if not p: return
        self.pl.screenshot(p)
        QMessageBox.information(self, "Export", f"PNG: {p}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    MainWin().show()
    print("VAE-FDM running.")
    print("Deterministic metrics match paper Table 1. VAE diversity is experimental.")
    app.exec()

if __name__ == "__main__":
    main()
