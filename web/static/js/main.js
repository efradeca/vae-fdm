/**
 * Main entry point for VAE-FDM web explorer.
 * Matches the desktop PyVista app as closely as possible.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DragControls } from 'three/addons/controls/DragControls.js';
import { MeshRenderer } from './mesh-renderer.js';
import { fetchTopology, predictDebounced } from './api-client.js';
import { colormapGradientCSS } from './colormap.js';

// ---------------------------------------------------------------------------
// Scene (white background like PyVista: white -> aliceblue)
// ---------------------------------------------------------------------------
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
camera.position.set(18, -18, 14);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// Lighting (match PyVista default)
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
dl1.position.set(10, -10, 15);
scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0xffffff, 0.25);
dl2.position.set(-10, 10, 5);
scene.add(dl2);

// Axes helper (like pl.add_axes())
const axes = new THREE.AxesHelper(2);
axes.position.set(-0.5, -0.5, -0.5);
scene.add(axes);

// Orbit controls
const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.1;
orbitControls.target.set(0, 0, 1);

const meshRenderer = new MeshRenderer(scene);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let topology = null;
let currentParams = {};
let colorMode = 'q';
let latestData = null;
let symmetryLocked = true;

// Draggable control point spheres
const cpSpheres = [];
let dragControls = null;

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------
function onResize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);

// ---------------------------------------------------------------------------
// Mirror a quarter-tile to full 16 control points (double symmetry).
// Matches calculate_grid_from_tile_quarter in neural_fdm/generators/grids.py:
//   1. Mirror across YZ plane (negate X)
//   2. Mirror the 8 points across XZ plane (negate Y)
// ---------------------------------------------------------------------------
function mirrorQuarterTile(tile4) {
    // Step 1: original + mirror across YZ (negate x)
    const step1 = [];
    for (const [x, y, z] of tile4) {
        step1.push([x, y, z]);
        step1.push([-x, y, z]);
    }
    // Step 2: step1 + mirror across XZ (negate y)
    const all = [];
    for (const [x, y, z] of step1) {
        all.push([x, y, z]);
        all.push([x, -y, z]);
    }
    return all;
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function buildUI(topo) {
    const { bounds, presets, tile } = topo;

    // Sliders
    const slidersDiv = document.getElementById('sliders');
    slidersDiv.innerHTML = '';
    for (const [key, b] of Object.entries(bounds)) {
        currentParams[key] = b.default;
        const row = document.createElement('div');
        row.className = 'slider-row';
        row.innerHTML = `
            <label>${b.label}</label>
            <input type="range" min="${b.min}" max="${b.max}" step="0.01"
                   value="${b.default}" data-key="${key}">
            <span class="val">${b.default.toFixed(2)}</span>
        `;
        slidersDiv.appendChild(row);

        const input = row.querySelector('input');
        const valSpan = row.querySelector('.val');
        input.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            currentParams[key] = v;
            valSpan.textContent = v.toFixed(2);
            requestPrediction();
        });
    }

    // Presets
    const presetsDiv = document.getElementById('presets');
    presetsDiv.innerHTML = '';
    for (const [name, params] of Object.entries(presets)) {
        const btn = document.createElement('button');
        btn.textContent = name;
        btn.addEventListener('click', () => {
            for (const [k, v] of Object.entries(params)) {
                currentParams[k] = v;
                const input = slidersDiv.querySelector(`input[data-key="${k}"]`);
                if (input) {
                    input.value = v;
                    input.parentElement.querySelector('.val').textContent = v.toFixed(2);
                }
            }
            requestPrediction();
        });
        presetsDiv.appendChild(btn);
    }

    // Break symmetry checkbox
    document.getElementById('break-symmetry').addEventListener('change', (e) => {
        symmetryLocked = !e.target.checked;
    });

    // Color mode
    document.getElementById('color-mode').addEventListener('change', (e) => {
        colorMode = e.target.value;
        const label = e.target.options[e.target.selectedIndex].text;
        document.getElementById('colorbar-title').textContent = label;
        if (latestData) updateView(latestData);
    });

    // Visibility toggles
    document.getElementById('show-target').addEventListener('change', (e) => {
        meshRenderer.targetGroup.visible = e.target.checked;
    });
    document.getElementById('show-surface').addEventListener('change', (e) => {
        meshRenderer.surfaceGroup.visible = e.target.checked;
    });
    document.getElementById('show-supports').addEventListener('change', (e) => {
        meshRenderer.supportsGroup.visible = e.target.checked;
    });
    document.getElementById('show-cp').addEventListener('change', (e) => {
        meshRenderer.cpGroup.visible = e.target.checked;
        meshRenderer.cpDragGroup.visible = e.target.checked;
        cpSpheres.forEach(s => s.visible = e.target.checked);
    });

    // Colorbar gradient
    document.getElementById('colorbar-gradient').style.background = colormapGradientCSS();

    // Build draggable control points
    buildControlPointSpheres(tile);
}

// ---------------------------------------------------------------------------
// 3D control point spheres
// ---------------------------------------------------------------------------
function buildControlPointSpheres(tile) {
    const cpGeo = new THREE.SphereGeometry(0.2, 16, 16);

    for (let i = 0; i < 4; i++) {
        const mat = new THREE.MeshPhongMaterial({ color: 0xff0000 });
        const sphere = new THREE.Mesh(cpGeo, mat);
        const base = tile[i];
        sphere.userData = { cpIndex: i, base };
        scene.add(sphere);
        cpSpheres.push(sphere);
    }
    updateControlPointPositions();

    dragControls = new DragControls(cpSpheres, camera, renderer.domElement);
    dragControls.addEventListener('dragstart', () => { orbitControls.enabled = false; });
    dragControls.addEventListener('dragend', () => { orbitControls.enabled = true; });
    dragControls.addEventListener('drag', (event) => {
        const obj = event.object;
        const { cpIndex, base } = obj.userData;

        if (symmetryLocked) {
            // Constrain to symmetric DOFs only (match desktop sliders)
            if (cpIndex === 0) {
                // Only z movement
                currentParams.c1_z = clamp(obj.position.z - base[2], 1, 10);
                obj.position.x = base[0];
                obj.position.y = base[1];
                obj.position.z = base[2] + currentParams.c1_z;
            } else if (cpIndex === 1) {
                // x and z movement
                currentParams.c2_x = clamp(obj.position.x - base[0], -5, 5);
                currentParams.c2_z = clamp(obj.position.z - base[2], 0, 10);
                obj.position.y = base[1];
                obj.position.x = base[0] + currentParams.c2_x;
                obj.position.z = base[2] + currentParams.c2_z;
            } else if (cpIndex === 2) {
                // Only y movement
                currentParams.c3_y = clamp(obj.position.y - base[1], -5, 5);
                obj.position.x = base[0];
                obj.position.z = base[2];
                obj.position.y = base[1] + currentParams.c3_y;
            } else {
                // c4 is fixed
                obj.position.set(base[0], base[1], base[2]);
            }
        } else {
            // Free drag (break symmetry) - still map back to nearest params
            if (cpIndex === 0) {
                currentParams.c1_z = clamp(obj.position.z - base[2], 1, 10);
            } else if (cpIndex === 1) {
                currentParams.c2_x = clamp(obj.position.x - base[0], -5, 5);
                currentParams.c2_z = clamp(obj.position.z - base[2], 0, 10);
            } else if (cpIndex === 2) {
                currentParams.c3_y = clamp(obj.position.y - base[1], -5, 5);
            }
        }
        syncSlidersFromParams();
        requestPrediction();
    });
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function syncSlidersFromParams() {
    const slidersDiv = document.getElementById('sliders');
    for (const [k, v] of Object.entries(currentParams)) {
        const input = slidersDiv.querySelector(`input[data-key="${k}"]`);
        if (input) {
            input.value = v;
            input.parentElement.querySelector('.val').textContent = v.toFixed(2);
        }
    }
}

function updateControlPointPositions() {
    if (!topology) return;
    const tile = topology.tile;

    // Compute transformed tile (tile + transform), matching desktop:
    // transform = [[0,0,c1_z], [c2_x,0,c2_z], [0,c3_y,0], [0,0,0]]
    const cp4 = [
        [tile[0][0], tile[0][1], tile[0][2] + currentParams.c1_z],
        [tile[1][0] + currentParams.c2_x, tile[1][1], tile[1][2] + currentParams.c2_z],
        [tile[2][0], tile[2][1] + currentParams.c3_y, tile[2][2]],
        [tile[3][0], tile[3][1], tile[3][2]],
    ];

    // Update draggable red spheres
    cpSpheres.forEach((s, i) => {
        s.position.set(cp4[i][0], cp4[i][1], cp4[i][2]);
    });

    // Update mirrored orange dots (all 16 points)
    const allCp = mirrorQuarterTile(cp4);
    meshRenderer.updateControlPoints(allCp);
}

// ---------------------------------------------------------------------------
// Prediction & view update
// ---------------------------------------------------------------------------
function requestPrediction() {
    stopDiversityAnimation();
    if (meshRenderer.resetSurfaceTint) meshRenderer.resetSurfaceTint();
    predictDebounced(currentParams, (data) => {
        latestData = data;
        updateView(data);
        updateControlPointPositions();
    });
}

function updateView(data) {
    const range = meshRenderer.update(data, colorMode);
    if (range) {
        // Show actual data range (like PyVista desktop), not symmetric
        const { vmin, vmax } = range;
        const fmt = Math.max(Math.abs(vmin), Math.abs(vmax)) < 0.01
            ? (v) => v.toExponential(1) : (v) => v.toFixed(3);
        document.getElementById('cb-max').textContent = fmt(vmax);
        document.getElementById('cb-mid1').textContent = fmt((vmin + vmax) * 0.75 + vmin * 0.25);
        document.getElementById('cb-zero').textContent = fmt((vmin + vmax) / 2);
        document.getElementById('cb-mid2').textContent = fmt(vmin * 0.75 + vmax * 0.25);
        document.getElementById('cb-min').textContent = fmt(vmin);
    }
    updateMetrics(data);

    const sel = document.getElementById('color-mode');
    document.getElementById('colorbar-title').textContent = sel.options[sel.selectedIndex].text;
}

function updateMetrics(data) {
    const { q, forces, inference_ms } = data;
    const qMin = Math.min(...q).toFixed(3);
    const qMax = Math.max(...q).toFixed(3);
    const fMin = Math.min(...forces).toFixed(2);
    const fMax = Math.max(...forces).toFixed(2);
    const allComp = q.every(v => v <= 0.001);

    document.getElementById('metrics').innerHTML = `
        Inference: <span class="value">${inference_ms} ms</span><br>
        q range: <span class="value">[${qMin}, ${qMax}]</span><br>
        F range: <span class="value">[${fMin}, ${fMax}]</span><br>
        Edges: <span class="value">${q.length}</span><br>
        All compression: <span class="${allComp ? 'ok' : 'value'}">${allComp ? 'Yes' : 'No'}</span>
    `;
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function animate() {
    requestAnimationFrame(animate);
    orbitControls.update();
    renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// VAE Diversity — auto animation matching the desktop designer
// ---------------------------------------------------------------------------
let diversityFrames = null;
let diversityTimer = null;
let diversityTotal = 0;
let diversityIdx = 0;

const FRAME_INTERVAL_MS = 300;

async function checkVAE() {
    try {
        const res = await fetch('/api/has_vae');
        const data = await res.json();
        if (data.has_vae) {
            document.getElementById('vae-group').style.display = 'block';
            document.getElementById('diversity-panel').style.display = 'flex';
            initDiversityChartPlaceholder();
            document.getElementById('btn-diversity').addEventListener('click', requestDiversity);
        }
    } catch (e) { /* no VAE */ }
}

function initDiversityChartPlaceholder() {
    if (typeof Plotly === 'undefined') return;
    const el = document.getElementById('diversity-chart');
    if (!el) return;
    Plotly.newPlot(el, [], {
        title: { text: 'Click "Sample diverse equilibria" to begin', font: { size: 13 } },
        margin: { l: 52, r: 42, t: 40, b: 46 },
        paper_bgcolor: '#fcfcfc',
        plot_bgcolor: '#fafafa',
        xaxis: { title: 'Edge (sorted by design freedom)', zeroline: false },
        yaxis: { title: 'σ<sub>q</sub>', zeroline: false },
        font: { family: 'Segoe UI, system-ui, sans-serif', size: 11, color: '#222' },
    }, { displayModeBar: false, responsive: true });
}

let _diversityInflight = null;

async function requestDiversity() {
    const btn = document.getElementById('btn-diversity');

    // Guard against double-click races: cancel any in-flight fetch
    // *and* stop any running animation before issuing a new request.
    if (_diversityInflight) {
        try { _diversityInflight.abort(); } catch (_) {}
    }
    stopDiversityAnimation();
    diversityFrames = null;
    diversityIdx = 0;

    btn.textContent = 'Sampling...';
    btn.disabled = true;
    document.getElementById('diversity-status').textContent = 'Running VAE encoder on current target shape...';

    const ctrl = new AbortController();
    _diversityInflight = ctrl;
    const timeoutId = setTimeout(() => ctrl.abort(), 30000);

    try {
        const res = await fetch('/api/diversity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...currentParams, n_samples: 40 }),
            signal: ctrl.signal,
        });
        if (!res.ok) {
            let body = '';
            try { body = (await res.text()).slice(0, 200); } catch (_) {}
            throw new Error(`/api/diversity -> HTTP ${res.status}: ${body}`);
        }
        const data = await res.json();
        if (!data.samples) {
            throw new Error(data.error || 'No samples returned');
        }

        // Discard this response if another request superseded it mid-flight.
        if (_diversityInflight !== ctrl) return;

        const frames = data.samples.map(s => ({ ...s, kind: 'vae' }));
        frames.push({ ...data.deterministic, kind: 'det' });

        diversityFrames = frames;
        diversityTotal = frames.length;
        diversityIdx = 0;

        renderDiversityChart(data);
        btn.textContent = 'Resample';
        btn.disabled = false;

        diversityTimer = setInterval(tickDiversity, FRAME_INTERVAL_MS);
        tickDiversity();
    } catch (e) {
        if (e.name === 'AbortError') return;
        btn.textContent = 'Error sampling';
        btn.disabled = false;
        document.getElementById('diversity-status').textContent = 'Sampling failed: ' + e.message;
    } finally {
        clearTimeout(timeoutId);
        if (_diversityInflight === ctrl) _diversityInflight = null;
    }
}

function stopDiversityAnimation() {
    if (diversityTimer !== null) {
        clearInterval(diversityTimer);
        diversityTimer = null;
    }
}

function turboColor(frac) {
    // Compact 4-stop approximation of matplotlib's "turbo" colormap
    const stops = [
        [0.0, [0.18, 0.07, 0.34]],
        [0.3, [0.10, 0.58, 0.93]],
        [0.5, [0.31, 0.92, 0.40]],
        [0.7, [0.97, 0.76, 0.19]],
        [1.0, [0.77, 0.11, 0.05]],
    ];
    const x = Math.max(0, Math.min(1, frac));
    for (let i = 1; i < stops.length; i++) {
        if (x <= stops[i][0]) {
            const [x0, c0] = stops[i - 1];
            const [x1, c1] = stops[i];
            const t = (x - x0) / (x1 - x0 || 1);
            return [
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
            ];
        }
    }
    return stops[stops.length - 1][1];
}

function tickDiversity() {
    if (!diversityFrames || !latestData) return;
    const cur = diversityIdx;
    const frame = diversityFrames[cur];
    const isFinal = frame.kind === 'det';

    // Update mesh with the current frame's predicted vertices + q values
    const fakeData = {
        ...latestData,
        predicted: frame.predicted,
        q: frame.q,
        forces: frame.q.map((qi, i) => qi * (latestData.lengths ? latestData.lengths[i] : 1)),
    };
    meshRenderer.update(fakeData, colorMode);

    // Tint the surface: turbo color ramp for VAE samples, steelblue for final
    if (meshRenderer.setSurfaceTint) {
        if (isFinal) {
            meshRenderer.setSurfaceTint(0x1565c0, 0.85);
        } else {
            const frac = cur / Math.max(diversityTotal - 2, 1);
            const [r, g, b] = turboColor(frac);
            meshRenderer.setSurfaceTint((r * 255 << 16) | (g * 255 << 8) | (b * 255), 0.6);
        }
    }

    updateDiversityLine(cur, isFinal);

    // Status label
    const status = document.getElementById('diversity-status');
    if (isFinal) {
        status.innerHTML =
            `<b>Converged</b> — deterministic FDM prediction &nbsp;|&nbsp; shape error: ${frame.shape_error.toFixed(3)}`;
    } else {
        status.innerHTML =
            `Exploring <b>${cur + 1}/${diversityTotal - 1}</b> VAE samples — all satisfy mechanical equilibrium`;
    }

    diversityIdx++;
    if (isFinal) {
        stopDiversityAnimation();
    }
}

// ---------------------------------------------------------------------------
// Plotly chart: design-freedom bars + parallel-coordinates overlay
// ---------------------------------------------------------------------------
let chartSortIdx = null;
let chartQStd = null;
let chartXEdges = null;
let chartGhostCount = 0;

function renderDiversityChart(data) {
    if (typeof Plotly === 'undefined') return;
    const el = document.getElementById('diversity-chart');
    if (!el) return;

    chartSortIdx = data.sort_idx;
    chartQStd = chartSortIdx.map(i => data.q_std_per_edge[i]);
    chartXEdges = chartQStd.map((_, i) => i);
    chartGhostCount = 0;

    const bars = {
        type: 'bar',
        x: chartXEdges,
        y: chartQStd,
        marker: { color: '#f5b041', line: { width: 0 } },
        opacity: 0.55,
        name: 'Design freedom (population σ<sub>q</sub>)',
        yaxis: 'y',
        hoverinfo: 'skip',
    };

    // Ghost trace: grows as the animation progresses (we append per tick)
    const ghost = {
        type: 'scatter',
        mode: 'lines',
        x: [],
        y: [],
        line: { color: '#6b7280', width: 1.0 },
        opacity: 0.25,
        yaxis: 'y2',
        name: 'Past samples',
        hoverinfo: 'skip',
        connectgaps: false,
    };

    // Current iteration line (bright red, updates every tick)
    const current = {
        type: 'scatter',
        mode: 'lines',
        x: chartXEdges,
        y: new Array(chartXEdges.length).fill(null),
        line: { color: '#c62828', width: 2.6 },
        yaxis: 'y2',
        name: 'Current iteration',
        hoverinfo: 'skip',
    };

    // Final deterministic line (navy, drawn on last frame)
    const det = {
        type: 'scatter',
        mode: 'lines',
        x: chartXEdges,
        y: new Array(chartXEdges.length).fill(null),
        line: { color: '#0d47a1', width: 3.0 },
        yaxis: 'y2',
        name: 'Deterministic FDM',
        hoverinfo: 'skip',
        visible: true,
    };

    // y2 range derived from all q frames
    let qMin = Infinity, qMax = -Infinity;
    for (const f of diversityFrames) {
        for (const qi of f.q) {
            if (qi < qMin) qMin = qi;
            if (qi > qMax) qMax = qi;
        }
    }
    const qPad = 0.08 * Math.max(Math.abs(qMin), Math.abs(qMax), 1e-6);

    const layout = {
        title: {
            text: `Ensemble of ${diversityTotal - 1} equilibrium solutions  +  deterministic FDM`,
            font: { size: 12.5, color: '#1a1a1a' },
            x: 0.02, xanchor: 'left',
        },
        margin: { l: 58, r: 58, t: 46, b: 52 },
        paper_bgcolor: '#fcfcfc',
        plot_bgcolor: '#fafafa',
        font: { family: 'Segoe UI, system-ui, sans-serif', size: 11, color: '#222' },
        xaxis: {
            title: 'Edge (sorted by design freedom)',
            zeroline: false,
            showgrid: false,
            range: [-0.5, chartXEdges.length - 0.5],
        },
        yaxis: {
            title: 'σ<sub>q</sub> (population)',
            titlefont: { color: '#7a4f00' },
            tickfont: { color: '#7a4f00' },
            zeroline: false,
            gridcolor: '#e4e8ee',
        },
        yaxis2: {
            title: 'Force density q<sub>i</sub>',
            titlefont: { color: '#222' },
            tickfont: { color: '#222' },
            overlaying: 'y',
            side: 'right',
            zeroline: false,
            showgrid: false,
            range: [qMin - qPad, qMax + qPad],
        },
        showlegend: true,
        legend: {
            x: 0.99, y: 1.08, xanchor: 'right', yanchor: 'top',
            orientation: 'h',
            font: { size: 10 },
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: '#888',
            borderwidth: 0.6,
        },
    };

    // Purge before rebuild so ghost traces from prior runs don't
    // accumulate in memory across repeated samplings.
    try { Plotly.purge(el); } catch (_) {}
    Plotly.newPlot(el, [bars, ghost, current, det], layout,
        { displayModeBar: false, responsive: true });
}

function updateDiversityLine(idx, isFinal) {
    if (!chartSortIdx || typeof Plotly === 'undefined') return;
    const el = document.getElementById('diversity-chart');
    if (!el || !el.data) return;

    const q = diversityFrames[idx].q;
    const qOrdered = chartSortIdx.map(i => q[i]);

    if (!isFinal) {
        // Move previous current line into a cumulative ghost trace
        const prevData = el.data[2]; // index 2 = current line
        if (prevData && Array.isArray(prevData.y) && prevData.y.some(v => v !== null)) {
            const ghostX = (el.data[1].x || []).concat(prevData.x).concat([null]);
            const ghostY = (el.data[1].y || []).concat(prevData.y).concat([null]);
            Plotly.restyle(el, { x: [ghostX], y: [ghostY] }, [1]);
        }
        // Update current line
        Plotly.restyle(el, { y: [qOrdered] }, [2]);
        // Hide deterministic until final
        Plotly.restyle(el, { y: [new Array(qOrdered.length).fill(null)] }, [3]);
    } else {
        // Stash the last current line as ghost, clear current, draw deterministic
        const prevData = el.data[2];
        if (prevData && Array.isArray(prevData.y) && prevData.y.some(v => v !== null)) {
            const ghostX = (el.data[1].x || []).concat(prevData.x).concat([null]);
            const ghostY = (el.data[1].y || []).concat(prevData.y).concat([null]);
            Plotly.restyle(el, { x: [ghostX], y: [ghostY] }, [1]);
        }
        Plotly.restyle(el, { y: [new Array(qOrdered.length).fill(null)] }, [2]);
        Plotly.restyle(el, { y: [qOrdered] }, [3]);
    }
}

async function init() {
    onResize();
    topology = await fetchTopology();
    meshRenderer.init(topology);
    buildUI(topology);
    requestPrediction();
    checkVAE();
    animate();
}

init();
