/**
 * API client with debounced requests.
 */

let _timer = null;
let _topology = null;

/**
 * Fetch static topology (edges, bounds, presets) once.
 */
export async function fetchTopology() {
    if (_topology) return _topology;
    const res = await fetch('/api/topology');
    _topology = await res.json();
    return _topology;
}

/**
 * Request a prediction. Debounced to avoid flooding the server.
 */
export function predictDebounced(params, callback, delay = 50) {
    if (_timer) clearTimeout(_timer);
    _timer = setTimeout(async () => {
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            const data = await res.json();
            callback(data);
        } catch (e) {
            console.error('Predict failed:', e);
        }
    }, delay);
}
