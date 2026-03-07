/**
 * Clock — dual timezone (Brisbane + ET) with countdown support.
 *
 * Updates every second. Brisbane uses Australia/Brisbane timezone.
 * ET uses America/New_York (handles EST/EDT automatically via Intl).
 */

const BRIS_TZ = 'Australia/Brisbane';
const ET_TZ = 'America/New_York';

const brisFmt = new Intl.DateTimeFormat('en-AU', {
    timeZone: BRIS_TZ,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
});

const etFmt = new Intl.DateTimeFormat('en-US', {
    timeZone: ET_TZ,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
});

const brisDayFmt = new Intl.DateTimeFormat('en-AU', {
    timeZone: BRIS_TZ,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
});

let _intervalId = null;
let _brisEl = null;
let _etEl = null;
let _countdownEl = null;
let _countdownLabelEl = null;
let _nextSessionDt = null;
let _nextSessionName = null;
let _showCountdown = false;

/**
 * Format MM:SS countdown from total seconds.
 */
function formatCountdown(totalSeconds) {
    if (totalSeconds <= 0) return '00:00';
    const m = Math.floor(totalSeconds / 60);
    const s = Math.floor(totalSeconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/**
 * Single tick — update clocks and countdown.
 */
function tick() {
    const now = new Date();

    if (_brisEl) {
        _brisEl.textContent = brisFmt.format(now).toUpperCase();
    }

    if (_etEl) {
        _etEl.textContent = etFmt.format(now).toUpperCase();
    }

    // Countdown (only in APPROACHING/ALERT)
    if (_showCountdown && _countdownEl && _nextSessionDt) {
        const diffMs = new Date(_nextSessionDt).getTime() - now.getTime();
        const diffSec = Math.max(0, Math.floor(diffMs / 1000));
        _countdownEl.textContent = formatCountdown(diffSec);

        if (_countdownLabelEl && _nextSessionName) {
            _countdownLabelEl.textContent = `until ${_nextSessionName}`;
        }
    }
}

/**
 * Initialize clocks. Call once on load.
 * @param {Object} elements - { brisEl, etEl }
 */
export function initClock({ brisEl, etEl }) {
    _brisEl = brisEl;
    _etEl = etEl;

    if (_intervalId) clearInterval(_intervalId);
    tick(); // Immediate first tick
    _intervalId = setInterval(tick, 1000);
}

/**
 * Set countdown target. Call when state changes to APPROACHING or ALERT.
 * @param {string|null} isoDatetime - ISO datetime string for next session start
 * @param {string|null} sessionName - e.g., "CME_REOPEN"
 * @param {HTMLElement|null} countdownEl - element to write countdown text into
 * @param {HTMLElement|null} labelEl - element for countdown label
 */
export function setCountdownTarget(isoDatetime, sessionName, countdownEl, labelEl) {
    _nextSessionDt = isoDatetime;
    _nextSessionName = sessionName;
    _countdownEl = countdownEl;
    _countdownLabelEl = labelEl;
    _showCountdown = !!(isoDatetime && countdownEl);
}

/**
 * Hide countdown (when leaving APPROACHING/ALERT).
 */
export function clearCountdown() {
    _showCountdown = false;
    _nextSessionDt = null;
    _nextSessionName = null;
    if (_countdownEl) _countdownEl.textContent = '';
    if (_countdownLabelEl) _countdownLabelEl.textContent = '';
}

/**
 * Get current Brisbane date string for display.
 */
export function getBrisbaneDateStr() {
    return brisDayFmt.format(new Date());
}

/**
 * Stop clock (cleanup).
 */
export function stopClock() {
    if (_intervalId) {
        clearInterval(_intervalId);
        _intervalId = null;
    }
}
