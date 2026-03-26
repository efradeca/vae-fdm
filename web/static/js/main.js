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
// Init
// ---------------------------------------------------------------------------
async function init() {
    onResize();
    topology = await fetchTopology();
    meshRenderer.init(topology);
    buildUI(topology);
    requestPrediction();
    animate();
}

init();
