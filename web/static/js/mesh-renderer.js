/**
 * Three.js mesh renderer for FDM structures.
 * Matches the desktop PyVista app visual style.
 */
import * as THREE from 'three';
import { mapScalarsToColors } from './colormap.js';

export class MeshRenderer {
    constructor(scene) {
        this.scene = scene;

        this.targetGroup = new THREE.Group();
        this.surfaceGroup = new THREE.Group();
        this.supportsGroup = new THREE.Group();
        this.edgesGroup = new THREE.Group();
        this.cpGroup = new THREE.Group();       // mirrored control points (orange)
        this.cpDragGroup = new THREE.Group();   // draggable control points (red)

        scene.add(this.targetGroup);
        scene.add(this.surfaceGroup);
        scene.add(this.supportsGroup);
        scene.add(this.edgesGroup);
        scene.add(this.cpGroup);
        scene.add(this.cpDragGroup);

        this._edgeGeo = null;
        this._surfaceGeo = null;
        this._targetGeo = null;
        this._topology = null;
    }

    init(topology) {
        this._topology = topology;
        const { edges, boundary, num_vertices, num_uv } = topology;

        // --- Structural edges (line_width=3 in desktop, but WebGL lines are 1px) ---
        // Use cylinders for thick edges
        const numEdges = edges.length;
        const edgePositions = new Float32Array(numEdges * 2 * 3);
        const edgeColors = new Float32Array(numEdges * 2 * 3);
        edgeColors.fill(0.5);

        this._edgeGeo = new THREE.BufferGeometry();
        this._edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));
        this._edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3));

        // Note: WebGL ignores linewidth > 1 on most platforms.
        // For thicker edges, we render a second set of slightly offset lines.
        const edgeMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
        const edgeMesh = new THREE.LineSegments(this._edgeGeo, edgeMat);
        this.edgesGroup.add(edgeMesh);

        // --- Predicted surface (steelblue, opacity 0.4) ---
        const nu = num_uv;
        const surfPos = new Float32Array(num_vertices * 3);
        const indices = [];
        for (let i = 0; i < nu - 1; i++) {
            for (let j = 0; j < nu - 1; j++) {
                const a = i * nu + j;
                const b = (i + 1) * nu + j;
                const c = (i + 1) * nu + j + 1;
                const d = i * nu + j + 1;
                indices.push(a, b, c, a, c, d);
            }
        }

        this._surfaceGeo = new THREE.BufferGeometry();
        this._surfaceGeo.setAttribute('position', new THREE.BufferAttribute(surfPos, 3));
        this._surfaceGeo.setIndex(indices);
        this._surfaceGeo.computeVertexNormals();

        const surfMat = new THREE.MeshPhongMaterial({
            color: 0x4682b4,
            transparent: true,
            opacity: 0.4,
            side: THREE.DoubleSide,
            depthWrite: false,
            shininess: 30,
        });
        this.surfaceGroup.add(new THREE.Mesh(this._surfaceGeo, surfMat));

        // --- Target wireframe (gray, opacity 0.3) ---
        this._targetGeo = new THREE.BufferGeometry();
        const targetPos = new Float32Array(num_vertices * 3);
        this._targetGeo.setAttribute('position', new THREE.BufferAttribute(targetPos, 3));

        const wireIndices = [];
        for (let i = 0; i < nu; i++) {
            for (let j = 0; j < nu; j++) {
                const idx = i * nu + j;
                if (j < nu - 1) wireIndices.push(idx, idx + 1);
                if (i < nu - 1) wireIndices.push(idx, idx + nu);
            }
        }
        this._targetGeo.setIndex(wireIndices);

        const wireMat = new THREE.LineBasicMaterial({ color: 0x999999, transparent: true, opacity: 0.3 });
        this.targetGroup.add(new THREE.LineSegments(this._targetGeo, wireMat));

        // --- Support spheres (red, matching desktop) ---
        const sphereGeo = new THREE.SphereGeometry(0.1, 10, 10);
        const sphereMat = new THREE.MeshPhongMaterial({ color: 0xff3333 });
        boundary.forEach(() => {
            this.supportsGroup.add(new THREE.Mesh(sphereGeo, sphereMat));
        });
    }

    update(data, colorMode) {
        if (!this._topology) return;
        const { edges, boundary } = this._topology;
        const { target, predicted, q, forces } = data;

        const scalars = colorMode === 'forces' ? forces : q;

        // Edge positions
        const posArr = this._edgeGeo.attributes.position.array;
        for (let i = 0; i < edges.length; i++) {
            const [u, v] = edges[i];
            const pu = predicted[u], pv = predicted[v];
            const off = i * 6;
            posArr[off] = pu[0]; posArr[off + 1] = pu[1]; posArr[off + 2] = pu[2];
            posArr[off + 3] = pv[0]; posArr[off + 4] = pv[1]; posArr[off + 5] = pv[2];
        }
        this._edgeGeo.attributes.position.needsUpdate = true;

        // Edge colors from scalars
        const vmin = Math.min(...scalars);
        const vmax = Math.max(...scalars);
        const edgeColors = mapScalarsToColors(scalars, vmin, vmax);
        const colArr = this._edgeGeo.attributes.color.array;
        for (let i = 0; i < edges.length; i++) {
            const r = edgeColors[i * 3], g = edgeColors[i * 3 + 1], b = edgeColors[i * 3 + 2];
            const off = i * 6;
            colArr[off] = r; colArr[off + 1] = g; colArr[off + 2] = b;
            colArr[off + 3] = r; colArr[off + 4] = g; colArr[off + 5] = b;
        }
        this._edgeGeo.attributes.color.needsUpdate = true;

        // Surface
        const surfPos = this._surfaceGeo.attributes.position.array;
        for (let i = 0; i < predicted.length; i++) {
            surfPos[i * 3] = predicted[i][0];
            surfPos[i * 3 + 1] = predicted[i][1];
            surfPos[i * 3 + 2] = predicted[i][2];
        }
        this._surfaceGeo.attributes.position.needsUpdate = true;
        this._surfaceGeo.computeVertexNormals();

        // Target
        const tgtPos = this._targetGeo.attributes.position.array;
        for (let i = 0; i < target.length; i++) {
            tgtPos[i * 3] = target[i][0];
            tgtPos[i * 3 + 1] = target[i][1];
            tgtPos[i * 3 + 2] = target[i][2];
        }
        this._targetGeo.attributes.position.needsUpdate = true;

        // Supports
        const supports = this.supportsGroup.children;
        boundary.forEach((vi, idx) => {
            if (idx < supports.length) {
                supports[idx].position.set(predicted[vi][0], predicted[vi][1], predicted[vi][2]);
            }
        });

        return { vmin, vmax };
    }

    /**
     * Update mirrored control points (orange dots, matching desktop).
     * allCp is array of [x,y,z] for ALL 16 mirrored control points.
     */
    updateControlPoints(allCp) {
        // Remove old
        while (this.cpGroup.children.length) this.cpGroup.remove(this.cpGroup.children[0]);

        const geo = new THREE.SphereGeometry(0.15, 8, 8);
        const mat = new THREE.MeshPhongMaterial({ color: 0xff8c00, transparent: true, opacity: 0.5 });
        for (const pt of allCp) {
            const s = new THREE.Mesh(geo, mat);
            s.position.set(pt[0], pt[1], pt[2]);
            this.cpGroup.add(s);
        }
    }
}
