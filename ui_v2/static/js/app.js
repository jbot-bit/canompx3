/**
 * App — Main application entry point for the ORB Trading Cockpit V2.
 *
 * On load:
 * 1. Fetch /api/state -> determine current panel
 * 2. Fetch /api/briefings -> populate briefing cards
 * 3. Fetch /api/rolling-pnl -> populate sparkline area
 * 4. Start clock interval
 * 5. Connect to /api/events SSE (graceful fallback to polling)
 * 6. Set up keyboard shortcuts
 *
 * Security note: All innerHTML usage in this file renders data from our own
 * trusted FastAPI backend (/api/* endpoints). This is a single-user local
 * trading dashboard, not a public-facing application. No user-supplied or
 * third-party content reaches innerHTML without being constructed by this code.
 */

import { SSEClient } from './sse-client.js';
import { initClock, setCountdownTarget, clearCountdown, getBrisbaneDateStr } from './clock.js';
import { initKeyboard, buildOverlayHTML } from './keyboard.js';

// ── State ────────────────────────────────────────────────────────────────

let currentState = 'IDLE';
let stateData = null;
let briefingsData = [];
let rollingPnlData = null;
let sseClient = null;

const PANEL_IDS = [
    'panel-weekend',
    'panel-overnight',
    'panel-idle',
    'panel-approaching',
    'panel-alert',
    'panel-orb-forming',
    'panel-in-session',
    'panel-debrief',
];

const STATE_TO_PANEL = {
    'WEEKEND':      'panel-weekend',
    'OVERNIGHT':    'panel-overnight',
    'IDLE':         'panel-idle',
    'APPROACHING':  'panel-approaching',
    'ALERT':        'panel-alert',
    'ORB_FORMING':  'panel-orb-forming',
    'IN_SESSION':   'panel-in-session',
    'DEBRIEF':      'panel-debrief',
};

// ── DOM Helpers ──────────────────────────────────────────────────────────

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

/** Create an element with optional classes and textContent */
function createElement(tag, { className, text, mono } = {}) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text) el.textContent = text;
    if (mono) el.style.fontFamily = 'var(--font-mono)';
    return el;
}

/** Safely escape a string for display (defense-in-depth) */
function esc(str) {
    if (str === null || str === undefined) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── API Fetchers ─────────────────────────────────────────────────────────

async function fetchJSON(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`[App] Fetch error for ${url}:`, err);
        return null;
    }
}

async function fetchState() {
    const data = await fetchJSON('/api/state');
    if (data) {
        stateData = data;
        applyState(data.name, data);
    }
}

async function fetchBriefings() {
    const data = await fetchJSON('/api/briefings');
    if (data && data.briefings) {
        briefingsData = data.briefings;
        renderBriefings(data.briefings);
    }
}

async function fetchRollingPnl() {
    const data = await fetchJSON('/api/rolling-pnl');
    if (data) {
        rollingPnlData = data;
        renderSparkline(data);
    }
}

async function fetchDaySummary() {
    const data = await fetchJSON('/api/day-summary');
    if (data) {
        renderDaySummary(data);
    }
}

async function fetchFitness() {
    const data = await fetchJSON('/api/fitness');
    if (data && data.strategies) {
        renderFitnessSummary(data.strategies);
    }
}

// ── Panel Switching ──────────────────────────────────────────────────────

function applyState(stateName, data) {
    currentState = stateName;

    // Set data-state on #app for CSS state selectors
    document.getElementById('app').setAttribute('data-state', stateName);

    // Hide all panels, show the active one
    for (const id of PANEL_IDS) {
        const el = document.getElementById(id);
        if (!el) continue;
        if (id === STATE_TO_PANEL[stateName]) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    }

    // Update statusbar
    updateStatusbar(stateName, data);

    // Countdown management
    if (stateName === 'APPROACHING' || stateName === 'ALERT') {
        const countdownEl = stateName === 'ALERT'
            ? $('#alert-countdown-value')
            : $('#approaching-countdown-value');
        const labelEl = stateName === 'ALERT'
            ? $('#alert-countdown-label')
            : $('#approaching-countdown-label');

        setCountdownTarget(
            data.next_session_dt,
            data.next_session,
            countdownEl,
            labelEl
        );

        // Show session name
        const nameEl = stateName === 'ALERT'
            ? $('.alert__session-name')
            : $('.approaching__session-name');
        if (nameEl && data.next_session) {
            nameEl.textContent = data.next_session;
        }
    } else {
        clearCountdown();
    }

    // IDLE-specific: show next session time (not countdown)
    if (stateName === 'IDLE' && data) {
        renderIdlePanel(data);
    }

    // Weekend-specific
    if (stateName === 'WEEKEND' && data) {
        renderWeekendPanel(data);
    }

    // Overnight-specific
    if (stateName === 'OVERNIGHT' && data) {
        renderOvernightPanel(data);
    }

    // Update session strip dots
    updateSessionStrip(data);

    // Connection indicator
    updateConnectionIndicator();
}

// ── Statusbar ────────────────────────────────────────────────────────────

function updateStatusbar(stateName, data) {
    const stateEl = $('#statusbar__state');
    const nextEl = $('#statusbar__next');
    const dailyEl = $('#statusbar__daily');

    if (stateEl) {
        stateEl.textContent = `State: ${stateName}`;
    }

    if (nextEl && data) {
        if (data.next_session) {
            const mins = data.minutes_to_next;
            if (mins !== null && mins !== undefined) {
                if (mins < 60) {
                    nextEl.textContent = `Next: ${data.next_session} (${Math.round(mins)}m)`;
                } else {
                    const h = Math.floor(mins / 60);
                    const m = Math.round(mins % 60);
                    nextEl.textContent = `Next: ${data.next_session} (${h}h ${m}m)`;
                }
            } else {
                nextEl.textContent = `Next: ${data.next_session}`;
            }
        } else {
            nextEl.textContent = '';
        }
    }

    if (dailyEl && rollingPnlData) {
        const todayR = rollingPnlData.today_r;
        if (todayR !== undefined && todayR !== null) {
            const sign = todayR >= 0 ? '+' : '';
            dailyEl.textContent = `Daily: ${sign}${todayR.toFixed(1)}R`;
            dailyEl.style.color = todayR >= 0 ? 'var(--color-long)' : 'var(--color-short)';
        }
    }
}

// ── Session Strip ────────────────────────────────────────────────────────

function updateSessionStrip(data) {
    const strip = $('#session-strip');
    if (!strip) return;

    // Clear existing dots
    while (strip.firstChild) strip.removeChild(strip.firstChild);

    const dotCount = Math.max(briefingsData.length, 5);

    for (let i = 0; i < dotCount; i++) {
        const dot = document.createElement('div');
        dot.className = 'session-dot session-dot--upcoming';

        if (i < briefingsData.length) {
            const b = briefingsData[i];
            dot.setAttribute('data-tooltip', b.session);

            if (data && data.next_session === b.session && currentState === 'APPROACHING') {
                dot.className = 'session-dot session-dot--approaching';
            } else if (data && data.next_session === b.session && currentState === 'ALERT') {
                dot.className = 'session-dot session-dot--active';
            }
        }

        strip.appendChild(dot);
    }
}

// ── Briefing Rendering (DOM-based) ───────────────────────────────────────

function createBriefingCard(b) {
    const card = createElement('div', { className: 'briefing-card' });

    // Header
    const header = createElement('div', { className: 'briefing-card__header' });
    header.appendChild(createElement('span', { className: 'briefing-card__session', text: b.session }));
    header.appendChild(createElement('span', { className: 'briefing-card__instrument', text: b.instrument }));
    if (b.orb_minutes) {
        header.appendChild(createElement('span', { className: 'briefing-card__instrument', text: `${b.orb_minutes}m` }));
    }
    card.appendChild(header);

    // RR detail
    card.appendChild(createDetailRow('RR', b.rr_target || '--'));

    // Entry detail
    card.appendChild(createDetailRow('Entry', b.entry_instruction || 'E2 stop-market'));

    // Direction
    if (b.direction_note) {
        card.appendChild(createDetailRow('Dir', b.direction_note));
    }

    // Filter/conditions
    if (b.conditions) {
        card.appendChild(createDetailRow('Filter', b.conditions));
    }

    // Strategy count
    const countEl = createElement('div', { className: 'briefing-card__strategies', text: `${b.strategy_count || 0} strategies` });
    card.appendChild(countEl);

    return card;
}

function createDetailRow(label, value) {
    const row = createElement('div', { className: 'briefing-card__detail' });
    row.appendChild(createElement('span', { className: 'briefing-card__label', text: label }));
    row.appendChild(createElement('span', { className: 'briefing-card__value', text: value }));
    return row;
}

function renderBriefings(briefings) {
    // Main panel containers
    const containers = [
        $('.approaching__briefings'),
        $('.alert__briefings'),
    ];

    for (const container of containers) {
        if (!container) continue;
        while (container.firstChild) container.removeChild(container.firstChild);

        if (briefings.length === 0) {
            const empty = createElement('div', { className: 'empty-state' });
            empty.appendChild(createElement('div', { className: 'empty-state__text', text: 'No upcoming sessions with active strategies' }));
            container.appendChild(empty);
        } else {
            for (const b of briefings) {
                container.appendChild(createBriefingCard(b));
            }
        }
    }

    // Sidebar briefing preview (first 3)
    const sidebarBriefings = $('#sidebar-briefings');
    if (sidebarBriefings) {
        while (sidebarBriefings.firstChild) sidebarBriefings.removeChild(sidebarBriefings.firstChild);

        if (briefings.length === 0) {
            sidebarBriefings.appendChild(
                createElement('div', { className: 'text-tertiary', text: 'No upcoming sessions' })
            );
        } else {
            for (const b of briefings.slice(0, 3)) {
                const card = createElement('div', { className: 'card' });
                card.style.padding = 'var(--space-3)';

                const row = createElement('div');
                row.style.cssText = 'display:flex;justify-content:space-between;align-items:center';

                const name = createElement('span', { className: 'mono', text: b.session });
                name.style.cssText = 'font-size:0.8125rem;font-weight:600';
                row.appendChild(name);

                const inst = createElement('span', { className: 'mono text-secondary', text: b.instrument });
                inst.style.fontSize = '0.75rem';
                row.appendChild(inst);

                card.appendChild(row);

                const detail = createElement('div', {
                    className: 'text-tertiary',
                    text: `${b.strategy_count || 0} strategies | RR ${b.rr_target || '--'}`
                });
                detail.style.cssText = 'font-size:0.75rem;margin-top:2px';
                card.appendChild(detail);

                sidebarBriefings.appendChild(card);
            }
        }
    }
}

// ── Sparkline Rendering (SVG, no innerHTML needed for data) ──────────────

function renderSparkline(data) {
    const container = $('#idle-sparkline');
    if (!container) return;

    const points = data.daily_points || [];
    if (points.length === 0) {
        while (container.firstChild) container.removeChild(container.firstChild);
        container.appendChild(createElement('div', { className: 'text-tertiary', text: 'No P&L data available' }));
        return;
    }

    const width = 400;
    const height = 50;
    const padding = 4;

    const values = points.map(p => p.cumulative_r || 0);
    const minVal = Math.min(0, ...values);
    const maxVal = Math.max(0, ...values);
    const range = maxVal - minVal || 1;

    const xStep = (width - padding * 2) / Math.max(values.length - 1, 1);

    const coords = values.map((v, i) => {
        const x = padding + i * xStep;
        const y = height - padding - ((v - minVal) / range) * (height - padding * 2);
        return { x, y };
    });

    const pathD = coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x.toFixed(1)} ${c.y.toFixed(1)}`).join(' ');
    const zeroY = height - padding - ((0 - minVal) / range) * (height - padding * 2);
    const finalVal = values[values.length - 1];
    const lineColor = finalVal >= 0 ? 'var(--color-long)' : 'var(--color-short)';

    // Build SVG via DOM API
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.style.cssText = 'width:100%;height:100%';

    // Zero line
    const zeroLine = document.createElementNS(NS, 'line');
    zeroLine.setAttribute('x1', padding);
    zeroLine.setAttribute('y1', zeroY.toFixed(1));
    zeroLine.setAttribute('x2', width - padding);
    zeroLine.setAttribute('y2', zeroY.toFixed(1));
    zeroLine.setAttribute('stroke', 'var(--surface-border)');
    zeroLine.setAttribute('stroke-width', '0.5');
    zeroLine.setAttribute('stroke-dasharray', '2,2');
    svg.appendChild(zeroLine);

    // Path
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', pathD);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', lineColor);
    path.setAttribute('stroke-width', '1.5');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);

    // End dot
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('cx', coords[coords.length - 1].x.toFixed(1));
    circle.setAttribute('cy', coords[coords.length - 1].y.toFixed(1));
    circle.setAttribute('r', '2.5');
    circle.setAttribute('fill', lineColor);
    svg.appendChild(circle);

    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(svg);

    // Summary below sparkline
    const summaryEl = $('#idle-sparkline-summary');
    if (summaryEl) {
        while (summaryEl.firstChild) summaryEl.removeChild(summaryEl.firstChild);

        const addStat = (label, val) => {
            if (val === undefined || val === null) return;
            const sign = val >= 0 ? '+' : '';
            const color = val >= 0 ? 'var(--color-long)' : 'var(--color-short)';

            const span = createElement('span', { text: `${label}: ` });
            span.style.cssText = 'font-size:0.8125rem;color:var(--text-secondary)';

            const valSpan = createElement('span', { text: `${sign}${val.toFixed(1)}R`, className: 'mono' });
            valSpan.style.color = color;
            span.appendChild(valSpan);

            if (summaryEl.childNodes.length > 0) {
                const sep = createElement('span', { className: 'statusbar__separator' });
                sep.style.cssText = 'display:inline-block;vertical-align:middle;margin:0 var(--space-3)';
                summaryEl.appendChild(sep);
            }
            summaryEl.appendChild(span);
        };

        addStat('Week', data.week_r);
        addStat('Month', data.month_r);
    }
}

// ── Idle Panel ───────────────────────────────────────────────────────────

function renderIdlePanel(data) {
    const timeEl = $('#idle-next-time');
    const nameEl = $('#idle-next-name');

    if (timeEl && data.next_session_dt) {
        const dt = new Date(data.next_session_dt);
        const timeFmt = new Intl.DateTimeFormat('en-AU', {
            timeZone: 'Australia/Brisbane',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true,
        });
        timeEl.textContent = timeFmt.format(dt).toUpperCase();
    }

    if (nameEl && data.next_session) {
        nameEl.textContent = data.next_session;
    }
}

// ── Weekend Panel ────────────────────────────────────────────────────────

function renderWeekendPanel(data) {
    const subtitleEl = $('.weekend__subtitle');
    if (subtitleEl && data.next_monday) {
        const monday = new Date(data.next_monday);
        const fmt = new Intl.DateTimeFormat('en-AU', {
            timeZone: 'Australia/Brisbane',
            weekday: 'long',
            month: 'long',
            day: 'numeric',
        });
        subtitleEl.textContent = `Markets reopen ${fmt.format(monday)}`;
    }
}

// ── Overnight Panel ──────────────────────────────────────────────────────

function renderOvernightPanel(data) {
    const subtitleEl = $('.overnight__subtitle');
    if (subtitleEl) {
        subtitleEl.textContent = 'Next session outside awake hours. Rest up.';
    }
}

// ── Day Summary ──────────────────────────────────────────────────────────

function renderDaySummary(data) {
    const el = $('#idle-completed-sessions');
    if (!el) return;

    while (el.firstChild) el.removeChild(el.firstChild);

    const sessions = data.sessions || [];
    if (sessions.length === 0) {
        el.appendChild(
            createElement('div', { className: 'text-tertiary', text: 'No completed sessions today' })
        );
        return;
    }

    for (const s of sessions) {
        const item = createElement('div', { className: 'session-history-item' });

        const left = createElement('span');
        const sessionName = createElement('span', { className: 'mono', text: s.session || s.orb_label || '--' });
        sessionName.style.cssText = 'font-size:0.8125rem;font-weight:500';
        left.appendChild(sessionName);

        if (s.instrument) {
            const inst = createElement('span', { className: 'text-tertiary', text: s.instrument });
            inst.style.cssText = 'font-size:0.75rem;margin-left:var(--space-2)';
            left.appendChild(inst);
        }
        item.appendChild(left);

        const pnl = s.pnl_r || 0;
        const resultClass = pnl >= 0 ? 'session-history-item__result--win' : 'session-history-item__result--loss';
        const sign = pnl >= 0 ? '+' : '';
        const result = createElement('span', {
            className: `mono session-history-item__result ${resultClass}`,
            text: `${sign}${pnl.toFixed(1)}R`
        });
        item.appendChild(result);

        el.appendChild(item);
    }
}

// ── Fitness Summary ──────────────────────────────────────────────────────

function renderFitnessSummary(strategies) {
    const el = $('#sidebar-fitness');
    if (!el) return;

    let fit = 0, watch = 0, decay = 0;
    for (const s of strategies) {
        const regime = (s.regime || s.fitness_regime || '').toUpperCase();
        if (regime === 'FIT') fit++;
        else if (regime === 'WATCH') watch++;
        else if (regime === 'DECAY') decay++;
    }

    while (el.firstChild) el.removeChild(el.firstChild);

    const grid = createElement('div', { className: 'fitness-grid' });

    const addCell = (count, label, colorVar) => {
        const cell = createElement('div', { className: 'fitness-grid__cell' });
        const countEl = createElement('div', { className: 'fitness-grid__count', text: String(count) });
        countEl.style.color = `var(${colorVar})`;
        const labelEl = createElement('div', { className: 'fitness-grid__label', text: label });
        labelEl.style.color = `var(${colorVar})`;
        cell.appendChild(countEl);
        cell.appendChild(labelEl);
        grid.appendChild(cell);
    };

    addCell(fit, 'FIT', '--regime-fit');
    addCell(watch, 'WATCH', '--regime-watch');
    addCell(decay, 'DECAY', '--regime-decay');

    el.appendChild(grid);
}

// ── Connection Indicator ─────────────────────────────────────────────────

function updateConnectionIndicator() {
    const el = $('#connection-indicator');
    if (!el) return;

    const labelEl = el.querySelector('.connection-indicator__label');
    if (sseClient && sseClient.isConnected) {
        el.className = 'connection-indicator connection-indicator--connected';
        if (labelEl) labelEl.textContent = 'Connected';
    } else {
        el.className = 'connection-indicator connection-indicator--reconnecting';
        if (labelEl) labelEl.textContent = 'Polling';
    }
}

// ── Checklist (APPROACHING/ALERT) ────────────────────────────────────────

function initChecklist() {
    const items = $$('.checklist__item');
    for (const item of items) {
        item.addEventListener('click', () => {
            item.classList.toggle('checklist__item--checked');
        });
    }
}

// ── Audio Toggle ─────────────────────────────────────────────────────────

function initAudioToggle() {
    const btn = $('#audio-toggle');
    if (!btn) return;

    const enabled = localStorage.getItem('audio_enabled') === 'true';
    btn.classList.toggle('audio-toggle--enabled', enabled);
    btn.textContent = enabled ? '\u{1F50A}' : '\u{1F507}';

    btn.addEventListener('click', () => {
        const next = !btn.classList.contains('audio-toggle--enabled');
        btn.classList.toggle('audio-toggle--enabled', next);
        btn.textContent = next ? '\u{1F50A}' : '\u{1F507}';
        localStorage.setItem('audio_enabled', String(next));
    });
}

// ── SSE Setup ────────────────────────────────────────────────────────────

function initSSE() {
    sseClient = new SSEClient('/api/events', {
        fallbackUrl: '/api/state',
        fallbackIntervalMs: 5000,
    });

    sseClient.on('state_change', (data) => {
        // SSE broadcasts use "current", polling fallback uses "name" — normalize
        const stateName = data.current || data.name;
        stateData = data;
        applyState(stateName, data);
    });

    sseClient.on('pnl_update', (data) => {
        if (data) {
            rollingPnlData = { ...rollingPnlData, ...data };
            updateStatusbar(currentState, stateData);
        }
    });

    sseClient.onStatusChange(() => {
        updateConnectionIndicator();
    });

    sseClient.connect();
}

// ── Init ─────────────────────────────────────────────────────────────────

async function init() {
    // Clock
    initClock({
        brisEl: $('#topbar__bris-time'),
        etEl: $('#topbar__et-time'),
    });

    // Keyboard shortcuts
    const overlayEl = $('#shortcut-overlay');
    if (overlayEl) {
        // Build overlay content via DOM (buildOverlayHTML returns trusted static content)
        const temp = document.createElement('div');
        temp.insertAdjacentHTML('beforeend', buildOverlayHTML());
        while (temp.firstChild) overlayEl.appendChild(temp.firstChild);
    }
    initKeyboard(overlayEl);

    // Audio toggle
    initAudioToggle();

    // Checklist click handlers
    initChecklist();

    // Date display
    const dateEl = $('#topbar__date');
    if (dateEl) {
        dateEl.textContent = getBrisbaneDateStr();
    }

    // Fetch initial data (parallel)
    await Promise.all([
        fetchState(),
        fetchBriefings(),
        fetchRollingPnl(),
        fetchDaySummary(),
        fetchFitness(),
    ]);

    // SSE connection (after initial data loaded)
    initSSE();
}

// Boot
document.addEventListener('DOMContentLoaded', init);
