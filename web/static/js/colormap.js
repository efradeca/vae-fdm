/**
 * Diverging colormap: coolwarm_r (reversed).
 * Negative values (compression) -> red/warm
 * Positive values (tension) -> blue/cool
 * Matches matplotlib coolwarm_r used in the desktop app.
 */

const STOPS = [
    [0.0, [0.706, 0.016, 0.150]],    // dark red (max compression)
    [0.25, [0.929, 0.567, 0.459]],
    [0.5, [0.865, 0.865, 0.865]],    // neutral gray (zero)
    [0.75, [0.557, 0.647, 0.925]],
    [1.0, [0.227, 0.298, 0.753]],    // dark blue (max tension)
];

function lerp(a, b, t) {
    return a + (b - a) * t;
}

export function colormapCoolwarmR(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 0; i < STOPS.length - 1; i++) {
        const [t0, c0] = STOPS[i];
        const [t1, c1] = STOPS[i + 1];
        if (t >= t0 && t <= t1) {
            const f = (t - t0) / (t1 - t0);
            return [
                lerp(c0[0], c1[0], f),
                lerp(c0[1], c1[1], f),
                lerp(c0[2], c1[2], f),
            ];
        }
    }
    return STOPS[STOPS.length - 1][1];
}

/**
 * Map scalars to colors using diverging colormap centered at 0.
 * Returns Float32Array of n*3 RGB values.
 *
 * This matches PyVista's behavior: symmetric range around 0,
 * so pure compression maps to the red half [0 -> 0.5].
 */
export function mapScalarsToColors(values, vmin, vmax) {
    const n = values.length;
    const colors = new Float32Array(n * 3);
    const absMax = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1e-6;

    for (let i = 0; i < n; i++) {
        // [-absMax, +absMax] -> [0, 1], with 0 -> 0.5
        const t = (values[i] / absMax + 1) * 0.5;
        const [r, g, b] = colormapCoolwarmR(t);
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
    return colors;
}

/**
 * Create CSS gradient for the colorbar.
 * Top = max (blue/tension), bottom = min (red/compression).
 */
export function colormapGradientCSS() {
    const stops = [];
    for (let i = 0; i <= 20; i++) {
        const t = 1 - i / 20;
        const [r, g, b] = colormapCoolwarmR(t);
        stops.push(`rgb(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0})`);
    }
    return `linear-gradient(to bottom, ${stops.join(', ')})`;
}
