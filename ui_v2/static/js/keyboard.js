/**
 * Keyboard — shortcut handler for the trading cockpit.
 *
 * Shortcuts are context-dependent (disabled when input focused).
 * Phase 3: ? (overlay), Esc (close). Others are placeholders for Phase 4-5.
 */

let _shortcuts = new Map();
let _overlayVisible = false;
let _overlayEl = null;
let _enabled = true;

/**
 * Built-in shortcut definitions.
 * Phase 3 implements ? and Esc. Others are registered as no-ops.
 */
const SHORTCUT_DEFS = [
    { key: '?',     label: 'Toggle shortcut help',       context: 'Any' },
    { key: 'Escape', label: 'Close expanded panel',      context: 'Any' },
    { key: ' ',      label: 'Clean Trade (1-click)',     context: 'DEBRIEF' },
    { key: 'd',      label: 'Expand/collapse debrief',   context: 'DEBRIEF' },
    { key: 'c',      label: 'Check all checklist items', context: 'APPROACHING / ALERT' },
    { key: '1',      label: 'Start Signal-Only mode',    context: 'ALERT' },
    { key: '2',      label: 'Start Demo mode',           context: 'ALERT' },
    { key: 's',      label: 'Stop session',              context: 'Session running' },
    { key: 'm',      label: 'Toggle manual trade log',   context: 'Signal-Only' },
    { key: 'ArrowLeft',  label: 'Previous session history', context: 'IDLE' },
    { key: 'ArrowRight', label: 'Next session history',     context: 'IDLE' },
];

/**
 * Register a keyboard shortcut handler.
 * @param {string} key - Key value (e.g., '?', 'Escape', ' ')
 * @param {Function} handler - Callback when shortcut fires
 */
export function registerShortcut(key, handler) {
    _shortcuts.set(key, handler);
}

/**
 * Initialize keyboard handler. Call once on load.
 * @param {HTMLElement} overlayEl - The shortcut overlay container
 */
export function initKeyboard(overlayEl) {
    _overlayEl = overlayEl;

    // Register built-in handlers
    registerShortcut('?', toggleOverlay);
    registerShortcut('Escape', handleEscape);

    document.addEventListener('keydown', _onKeyDown);
}

/**
 * Global keydown handler.
 */
function _onKeyDown(e) {
    if (!_enabled) return;

    // Ignore when typing in input/textarea/select
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
        // Allow Escape to blur from inputs
        if (e.key === 'Escape') {
            e.target.blur();
            return;
        }
        return;
    }

    const handler = _shortcuts.get(e.key);
    if (handler) {
        e.preventDefault();
        handler(e);
    }
}

/**
 * Toggle the shortcut help overlay.
 */
function toggleOverlay() {
    _overlayVisible = !_overlayVisible;
    if (_overlayEl) {
        _overlayEl.style.display = _overlayVisible ? 'flex' : 'none';
    }
}

/**
 * Handle Escape key — close overlay or other expanded panels.
 */
function handleEscape() {
    if (_overlayVisible) {
        _overlayVisible = false;
        if (_overlayEl) {
            _overlayEl.style.display = 'none';
        }
    }
    // Phase 4-5: close expanded debrief, modals, etc.
}

/**
 * Build the shortcut overlay HTML content.
 * @returns {string} HTML string
 */
export function buildOverlayHTML() {
    const rows = SHORTCUT_DEFS.map(def => {
        const keyDisplay = def.key === ' ' ? 'Space'
            : def.key === 'Escape' ? 'Esc'
            : def.key === 'ArrowLeft' ? String.fromCodePoint(0x2190)
            : def.key === 'ArrowRight' ? String.fromCodePoint(0x2192)
            : def.key;

        return `
            <div class="shortcut-overlay__row">
                <span>${def.label}</span>
                <span class="shortcut-overlay__key"><span class="kbd">${keyDisplay}</span>  <span class="text-tertiary" style="font-size:0.6875rem; margin-left:4px">${def.context}</span></span>
            </div>`;
    }).join('');

    return `
        <div class="shortcut-overlay__content">
            <div class="shortcut-overlay__title">Keyboard Shortcuts</div>
            ${rows}
            <div style="margin-top: var(--space-4); text-align: center;">
                <span class="text-tertiary" style="font-size: 0.75rem">Press <span class="kbd">?</span> or <span class="kbd">Esc</span> to close</span>
            </div>
        </div>`;
}

/**
 * Enable/disable all keyboard shortcuts.
 */
export function setKeyboardEnabled(enabled) {
    _enabled = enabled;
}

/**
 * Cleanup keyboard handler.
 */
export function destroyKeyboard() {
    document.removeEventListener('keydown', _onKeyDown);
    _shortcuts.clear();
}
