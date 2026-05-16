/**
 * API client with debounced requests, status-aware error reporting,
 * and AbortController-based cancellation of stale predictions.
 */

let _timer = null;
let _topology = null;
let _inflightController = null;

async function _checkOk(res, endpoint) {
    if (!res.ok) {
        let body = '';
        try { body = (await res.text()).slice(0, 200); } catch (_) {}
        throw new Error(`${endpoint} -> HTTP ${res.status}: ${body}`);
    }
}

/**
 * Fetch static topology (edges, bounds, presets) once.
 */
export async function fetchTopology() {
    if (_topology) return _topology;
    const res = await fetch('/api/topology');
    await _checkOk(res, '/api/topology');
    _topology = await res.json();
    return _topology;
}

/**
 * Request a prediction. Debounced to avoid flooding the server.
 * Cancels any in-flight prediction before issuing a new one.
 */
export function predictDebounced(params, callback, delay = 50) {
    if (_timer) clearTimeout(_timer);
    _timer = setTimeout(async () => {
        if (_inflightController) _inflightController.abort();
        _inflightController = new AbortController();
        const ctrl = _inflightController;
        const timeoutId = setTimeout(() => ctrl.abort(), 10000);
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
                signal: ctrl.signal,
            });
            await _checkOk(res, '/api/predict');
            const data = await res.json();
            callback(data);
        } catch (e) {
            if (e.name === 'AbortError') return;
            console.error('Predict failed:', e);
            const status = document.getElementById('diversity-status');
            if (status) status.textContent = 'Predict failed: ' + e.message;
        } finally {
            clearTimeout(timeoutId);
        }
    }, delay);
}
