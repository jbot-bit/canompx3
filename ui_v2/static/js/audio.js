/**
 * Audio Manager — Web Audio API tone generator for trading alerts.
 *
 * Muted by default. Toggle persisted in localStorage('audio_enabled').
 * Each alert is a short oscillator tone sequence.
 */

const STORAGE_KEY = "audio_enabled";
const VOLUME = 0.15;

let audioCtx = null;

function getContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playTone(ctx, frequency, type, startTime, duration, gainValue = VOLUME) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.value = frequency;
  gain.gain.value = gainValue;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(startTime);
  osc.stop(startTime + duration);
  return { osc, gain };
}

const ALERTS = {
  approaching(ctx) {
    // Soft single 440Hz sine, 200ms
    const now = ctx.currentTime;
    playTone(ctx, 440, "sine", now, 0.2);
  },

  alert(ctx) {
    // Two-tone ascending square: 660Hz then 880Hz, 150ms each, 50ms gap
    const now = ctx.currentTime;
    playTone(ctx, 660, "square", now, 0.15);
    playTone(ctx, 880, "square", now + 0.2, 0.15);
  },

  orb_complete(ctx) {
    // Three-note ascending sine: 440 -> 660 -> 880, 100ms each, 30ms gaps
    const now = ctx.currentTime;
    playTone(ctx, 440, "sine", now, 0.1);
    playTone(ctx, 660, "sine", now + 0.13, 0.1);
    playTone(ctx, 880, "sine", now + 0.26, 0.1);
  },

  signal_entry(ctx) {
    // Sharp ping: 880Hz sine, 100ms
    const now = ctx.currentTime;
    playTone(ctx, 880, "sine", now, 0.1);
  },

  signal_exit(ctx) {
    // Descending two-note sine: 660Hz then 440Hz, 150ms each, 50ms gap
    const now = ctx.currentTime;
    playTone(ctx, 660, "sine", now, 0.15);
    playTone(ctx, 440, "sine", now + 0.2, 0.15);
  },

  cooling(ctx) {
    // Low sustained 220Hz sine, 500ms with gain ramp down
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 220;
    gain.gain.setValueAtTime(VOLUME, now);
    gain.gain.linearRampToValueAtTime(0, now + 0.5);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.5);
  },
};

/**
 * Check localStorage and return whether audio is enabled.
 * Muted by default (returns false if no key set).
 * @returns {boolean}
 */
export function initAudio() {
  return localStorage.getItem(STORAGE_KEY) === "true";
}

/**
 * Play an alert tone by type. No-op if audio is muted.
 * @param {string} type — one of: approaching, alert, orb_complete,
 *   signal_entry, signal_exit, cooling
 */
export function playAlert(type) {
  if (!isAudioEnabled()) return;
  const handler = ALERTS[type];
  if (!handler) {
    console.warn(`[audio] Unknown alert type: ${type}`);
    return;
  }
  const ctx = getContext();
  handler(ctx);
}

/**
 * Return whether audio alerts are currently enabled.
 * @returns {boolean}
 */
export function isAudioEnabled() {
  return localStorage.getItem(STORAGE_KEY) === "true";
}

/**
 * Toggle audio on/off. Persists to localStorage.
 * @returns {boolean} New enabled state.
 */
export function toggleAudio() {
  const newState = !isAudioEnabled();
  localStorage.setItem(STORAGE_KEY, String(newState));
  return newState;
}
