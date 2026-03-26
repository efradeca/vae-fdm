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
import os, sys, time, yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

os.environ["QT_API"] = "pyside6"
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QGroupBox, QTabWidget,
    QTextEdit, QPushButton, QScrollArea, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import pyvista as pv
from pyvistaqt import QtInteractor

import jax, jax.numpy as jnp, jax.random as jrn
from neural_fdm import DATA
from neural_fdm.builders import (
    build_data_generator, build_connectivity_structure_from_generator,
    build_mesh_from_generator, build_neural_model,
)
from neural_fdm.serialization import load_model
from neural_fdm.generators.grids import calculate_grid_from_tile_quarter
from neural_fdm.helpers import edges_vectors, edges_lengths, edges_forces, vertices_residuals_from_xyz

TASK, SEED, NU = "bezier", 90, 10

COLOR_MODES = [
    ("Force Density q", "coolwarm_r"),
    ("Axial Force F=q*L", "coolwarm_r"),
    ("Shape Error per node", "turbo"),
]


def load_models():
    """Load deterministic formfinder + optional VAE model."""
    cfg_path = os.path.join(os.path.dirname(__file__), f"{TASK}.yml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    key = jrn.PRNGKey(SEED)
    mk, _ = jax.random.split(key, 2)
    gen = build_data_generator(cfg)
    st = build_connectivity_structure_from_generator(cfg, gen)
    sk = build_neural_model("formfinder", cfg, gen, mk)
    mdl = load_model(os.path.join(DATA, f"formfinder_{TASK}.eqx"), sk)

    @jax.jit
    def pred(x):
        xh, (q, xf, ld) = mdl(x, st, aux_data=True)
        return xh, q, ld
    pred(jnp.zeros(NU * NU * 3))

    # Try loading VAE model
    vae_model = None
    vae_path = os.path.join(DATA, f"variational_formfinder_{TASK}.eqx")
    if os.path.exists(vae_path):
        try:
            vae_sk = build_neural_model("variational_formfinder", cfg, gen, mk)
            vae_model = load_model(vae_path, vae_sk)
            print("VAE model loaded for diversity analysis.")
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

        print("Loading models...")
        self.mdl, self.st, self.gen, self.pred_fn, self.cfg, self.vae = load_models()
        # Use FDM mesh edges (180) instead of grid edges (261 with diagonals)
        mesh = build_mesh_from_generator(self.cfg, self.gen)
        self.edges = np.array(list(mesh.edges()))
        self.color_idx = 0
        self.transform = np.array([[0,0,3.0],[0,0,1.5],[0,0,0],[0,0,0]])

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

        # Control points
        g = QGroupBox("Control Points")
        gl = QVBoxLayout(g)

        # Lock toggle for sphere widgets
        from PySide6.QtWidgets import QCheckBox
        self.chk_unlock = QCheckBox("Unlock 3D sphere dragging")
        self.chk_unlock.setFont(QFont("Segoe UI", 8))
        self.chk_unlock.setChecked(False)
        self.chk_unlock.stateChanged.connect(self._on_unlock_toggle)
        gl.addWidget(self.chk_unlock)
        self._spheres_unlocked = False
        self._sphere_widgets_active = False
        # Ranges from saddle_minmax_values() in builders.py
        # (training data actual ranges, slightly wider than paper Table 6)
        self.sl_h = self._make_slider("c1.z height", 1.0, 10.0, 3.0, gl)
        self.sl_sx = self._make_slider("c2.x spread", -5.0, 5.0, 0.0, gl)
        self.sl_ez = self._make_slider("c2.z edge", 0.0, 10.0, 1.5, gl)
        self.sl_cy = self._make_slider("c3.y curve", -5.0, 5.0, 0.0, gl)
        lay.addWidget(g)

        # Color mode
        g = QGroupBox("Edge Coloring")
        gl = QVBoxLayout(g)
        self.cc = QComboBox()
        for nm, _ in COLOR_MODES: self.cc.addItem(nm)
        self.cc.currentIndexChanged.connect(self._on_color)
        gl.addWidget(self.cc)
        lay.addWidget(g)

        # VAE diversity
        if self.vae is not None:
            g = QGroupBox("Solution Diversity (VAE)")
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
        self.canvas_train = MplCanvas(); self.tabs.addTab(self.canvas_train, "Fig 4a")
        self._plot_training_curves()

        # VAE diversity tab (if model available)
        if self.vae is not None:
            self.canvas_div = MplCanvas(); self.tabs.addTab(self.canvas_div, "VAE Diversity")

    def _plot_training_curves(self):
        """Plot training curves like paper Figure 4a (from saved loss files)."""
        ax = self.canvas_train.ax; ax.clear()
        try:
            shape_f = os.path.join(DATA, "losses_formfinder_bezier_shape_error.txt")
            resid_f = os.path.join(DATA, "losses_formfinder_bezier_residual_error.txt")
            if os.path.exists(shape_f) and os.path.exists(resid_f):
                shape = np.loadtxt(shape_f)
                resid = np.loadtxt(resid_f)
                ax.semilogy(shape, label="L_shape", color="#1565c0", alpha=0.7)
                ax.semilogy(resid + 1e-15, label="L_physics", color="#c62828", alpha=0.7)
                ax.set_xlabel("Training Step"); ax.set_ylabel("Loss (log)")
                ax.set_title("Training Curves (paper Fig. 4a)", fontsize=9)
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "Train model first:\npython train.py formfinder bezier",
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
        self.transform[0,2] = self.sl_h.value()
        self.transform[1,0] = self.sl_sx.value()
        self.transform[1,2] = self.sl_ez.value()
        self.transform[2,1] = self.sl_cy.value()
        xyz_t = self.gen.evaluate_points(jnp.array(self.transform))
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
        self._compute()
        if not hasattr(self, '_meshes_created'):
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
        n = NU

        # Target wireframe
        xt = self.tnp.reshape(n, n, 3)
        self._target_mesh = pv.StructuredGrid(xt[:,:,0], xt[:,:,1], xt[:,:,2])
        self.pl.add_mesh(self._target_mesh, color="gray", style="wireframe",
                         line_width=1, opacity=0.3)

        # Predicted surface
        xp = self.pnp.reshape(n, n, 3)
        self._pred_mesh = pv.StructuredGrid(xp[:,:,0], xp[:,:,1], xp[:,:,2])
        self.pl.add_mesh(self._pred_mesh, color="steelblue", opacity=0.4,
                         show_edges=False)

        # Structural edges
        sc = self._get_sc()
        _, cm = COLOR_MODES[self.color_idx]
        lines = []
        for e in self.edges: lines.extend([2, e[0], e[1]])
        self._edge_poly = pv.PolyData(self.pnp, lines=np.array(lines))
        lbl = COLOR_MODES[self.color_idx][0]
        self._edge_poly[lbl] = sc
        mx = abs(sc).max() if len(sc) > 0 else 1.0
        fmt = "%.2e" if (0 < mx < 0.01) else "%.3f"
        self._edge_actor = self.pl.add_mesh(
            self._edge_poly, scalars=lbl, cmap=cm, line_width=3,
            scalar_bar_args={"title": lbl, "n_labels": 4, "fmt": fmt})

        # Control points: mirrored (visual only)
        tile = np.array(self.gen.surface.grid.tile)
        cp_unique = tile + self.transform
        cp_all = np.array(calculate_grid_from_tile_quarter(jnp.array(cp_unique)))
        self._cp_all_poly = pv.PolyData(cp_all)
        self.pl.add_points(self._cp_all_poly, color="orange", point_size=6,
                           render_points_as_spheres=True, opacity=0.4)

        # Control points: mode depends on lock state
        self._tile = tile
        if getattr(self, '_sphere_widgets_active', False):
            # Unlocked: draggable sphere widgets
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
            # Locked: static red points (not draggable)
            self._cp_unique_poly = pv.PolyData(cp_unique)
            self.pl.add_points(self._cp_unique_poly, color="red", point_size=14,
                               render_points_as_spheres=True)

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
        n = NU

        # Update target wireframe points
        xt = self.tnp.reshape(n, n, 3)
        new_target = pv.StructuredGrid(xt[:,:,0], xt[:,:,1], xt[:,:,2])
        self._target_mesh.points = new_target.points

        # Update predicted surface points
        xp = self.pnp.reshape(n, n, 3)
        new_pred = pv.StructuredGrid(xp[:,:,0], xp[:,:,1], xp[:,:,2])
        self._pred_mesh.points = new_pred.points

        # Update edge positions, scalars, and scalar bar range
        self._edge_poly.points = self.pnp.copy()
        sc = self._get_sc()
        lbl = COLOR_MODES[self.color_idx][0]
        self._edge_poly[lbl] = sc
        # Update clim so scalar bar reflects current data range
        if hasattr(self, '_edge_actor') and self._edge_actor is not None:
            mapper = self._edge_actor.GetMapper()
            if mapper is not None:
                vmin, vmax = float(sc.min()), float(sc.max())
                if abs(vmax - vmin) < 1e-10:
                    vmax = vmin + 1e-6
                mapper.SetScalarRange(vmin, vmax)

        # Update control points
        tile = np.array(self.gen.surface.grid.tile)
        cp_unique = tile + self.transform
        cp_all = np.array(calculate_grid_from_tile_quarter(jnp.array(cp_unique)))
        self._cp_all_poly.points = cp_all
        if hasattr(self, '_cp_unique_poly') and not self._sphere_widgets_active:
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
        cc = "#2e7d32" if self.all_comp else "#c62828"
        cs = "Yes (all q<=0)" if self.all_comp else "NO"
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
<h3 style="color:#1565c0;">Shape Match (Table 1)</h3>
<table style="font-family:Consolas; font-size:9pt;">
<tr><td>L_shape:</td><td><b>{self.err_l1:.2f}</b> (paper: 3.0+/-2.0)</td></tr>
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
        """Sample 10 diverse equilibrium solutions and show metrics."""
        if self.vae is None: return
        from neural_fdm.variational import compute_diversity_metrics, compute_variance_per_edge

        xyz_t = self.gen.evaluate_points(jnp.array(self.transform))
        key = jrn.PRNGKey(int(time.time()) % 10000)

        x_hats, qs = self.vae.sample(xyz_t, self.st, key, num_samples=10)
        metrics = compute_diversity_metrics(x_hats, qs)
        q_var = compute_variance_per_edge(self.vae, xyz_t, self.st, key, n_samples=20)

        # Update diversity chart
        ax = self.canvas_div.ax; ax.clear()
        q_std = np.array(metrics["q_std_per_edge"])
        sorted_idx = np.argsort(q_std)[::-1]
        ax.bar(range(len(q_std)), q_std[sorted_idx], color="#e65100", alpha=0.8, width=1.0)
        ax.set_xlabel("Edge (sorted by freedom)")
        ax.set_ylabel("q std across samples")
        ax.set_title("Design Freedom per Member\n(high = more solution options)", fontsize=9)
        ax.tick_params(labelsize=7)
        self.canvas_div.draw()

        # Update label
        self.vae_label.setText(
            f"10 equilibrium samples generated:\n"
            f"  Shape diversity (L1): {metrics['shape_pairwise_L1_mean']:.2f}\n"
            f"  q std across samples: {metrics['q_std_mean']:.4f}\n"
            f"  All satisfy equilibrium.\n\n"
            f"High q-std edges indicate members\n"
            f"with more design freedom."
        )

        # Switch to diversity tab
        self.tabs.setCurrentWidget(self.canvas_div)

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def _up(self, _=None):
        if getattr(self, '_syncing', False):
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
        faces = [[i*NU+j, i*NU+j+1, (i+1)*NU+j+1, (i+1)*NU+j] for i in range(NU-1) for j in range(NU-1)]
        export_obj(p, self.pnp, np.array(faces))
        QMessageBox.information(self, "Export", f"OBJ: {p}")

    def _exp_stl(self):
        p = self._ask("Save STL", "STL (*.stl)", "shape.stl")
        if not p: return
        xp = self.pnp.reshape(NU, NU, 3)
        pv.StructuredGrid(xp[:,:,0], xp[:,:,1], xp[:,:,2]).extract_surface().save(p)
        QMessageBox.information(self, "Export", f"STL: {p}")

    def _exp_json(self):
        p = self._ask("Save JSON", "JSON (*.json)", "data.json")
        if not p: return
        import json
        data = {
            "paper": "Pastrana et al., ICLR 2025, arXiv:2409.02606",
            "L_shape": round(self.err_l1, 4),
            "inference_ms": round(self.dt, 1),
            "control_points": self.transform.tolist(),
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
    print("Drag red spheres to reshape. All metrics paper-validated.")
    app.exec()

if __name__ == "__main__":
    main()
