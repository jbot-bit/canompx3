/**
 * SSE Client — EventSource wrapper with auto-reconnect and polling fallback.
 *
 * Connects to /api/events for real-time state updates. If SSE is unavailable
 * (Phase 2 not wired yet), falls back to polling /api/state.
 */

export class SSEClient {
    constructor(url, { fallbackUrl = '/api/state', fallbackIntervalMs = 5000 } = {}) {
        this._url = url;
        this._fallbackUrl = fallbackUrl;
        this._fallbackIntervalMs = fallbackIntervalMs;
        this._source = null;
        this._handlers = new Map();
        this._reconnectAttempts = 0;
        this._maxReconnectDelay = 30000;
        this._reconnectTimer = null;
        this._pollTimer = null;
        this._connected = false;
        this._usePolling = false;
        this._onStatusChange = null;
    }

    /** Register a status change callback: (status: 'connected'|'reconnecting'|'polling'|'disconnected') => void */
    onStatusChange(cb) {
        this._onStatusChange = cb;
    }

    /** Register handler for an SSE event type */
    on(eventType, callback) {
        if (!this._handlers.has(eventType)) {
            this._handlers.set(eventType, []);
        }
        this._handlers.get(eventType).push(callback);
    }

    /** Start connection — tries SSE first, falls back to polling */
    connect() {
        this._stopPolling();
        this._clearReconnectTimer();

        try {
            this._source = new EventSource(this._url);

            this._source.onopen = () => {
                this._connected = true;
                this._usePolling = false;
                this._reconnectAttempts = 0;
                this._emitStatus('connected');
            };

            this._source.onerror = () => {
                this._connected = false;
                this._source.close();
                this._source = null;

                // If first attempt fails, SSE endpoint likely not available — use polling
                if (this._reconnectAttempts === 0) {
                    this._startPolling();
                } else {
                    this._scheduleReconnect();
                }
            };

            // Listen for all registered event types
            this._source.addEventListener('message', (event) => {
                this._dispatch('message', event.data);
            });

            // Named SSE events
            for (const eventType of this._handlers.keys()) {
                if (eventType === 'message') continue;
                this._source.addEventListener(eventType, (event) => {
                    this._dispatch(eventType, event.data);
                });
            }
        } catch (e) {
            // EventSource not supported or URL invalid — fall back
            this._startPolling();
        }
    }

    /** Close connection and stop all timers */
    close() {
        if (this._source) {
            this._source.close();
            this._source = null;
        }
        this._stopPolling();
        this._clearReconnectTimer();
        this._connected = false;
        this._emitStatus('disconnected');
    }

    /** Whether we have an active connection (SSE or polling) */
    get isConnected() {
        return this._connected || this._usePolling;
    }

    // ── Internal ─────────────────────────────────────────────────────────

    _dispatch(eventType, rawData) {
        const handlers = this._handlers.get(eventType);
        if (!handlers) return;

        let data;
        try {
            data = JSON.parse(rawData);
        } catch {
            data = rawData;
        }

        for (const handler of handlers) {
            try {
                handler(data);
            } catch (err) {
                console.error(`[SSE] Handler error for '${eventType}':`, err);
            }
        }
    }

    _scheduleReconnect() {
        this._clearReconnectTimer();
        this._reconnectAttempts++;

        // Exponential backoff: 1s, 2s, 4s, 8s, ... max 30s
        const delay = Math.min(
            1000 * Math.pow(2, this._reconnectAttempts - 1),
            this._maxReconnectDelay
        );

        this._emitStatus('reconnecting');

        this._reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    _clearReconnectTimer() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
    }

    _startPolling() {
        this._usePolling = true;
        this._emitStatus('polling');
        this._poll(); // First poll immediately
        this._pollTimer = setInterval(() => this._poll(), this._fallbackIntervalMs);
    }

    _stopPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
        this._usePolling = false;
    }

    async _poll() {
        try {
            const resp = await fetch(this._fallbackUrl);
            if (!resp.ok) return;
            const data = await resp.json();

            // Dispatch as a 'state_change' event for compatibility
            const handlers = this._handlers.get('state_change');
            if (handlers) {
                for (const handler of handlers) {
                    try {
                        handler(data);
                    } catch (err) {
                        console.error('[SSE/Poll] Handler error:', err);
                    }
                }
            }
        } catch {
            // Network error during poll — ignore, will retry
        }
    }

    _emitStatus(status) {
        if (this._onStatusChange) {
            this._onStatusChange(status);
        }
    }
}
