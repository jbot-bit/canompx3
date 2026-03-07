/**
 * Signal Log — scrolling color-coded event log.
 *
 * Max 50 entries, auto-scrolls to bottom, color-coded by type.
 */

const MAX_ENTRIES = 50;

const TYPE_COLORS = {
  SIGNAL_ENTRY: 'var(--color-long)',
  SIGNAL_EXIT: 'var(--color-short)',
  MANUAL_ENTRY: 'var(--color-neutral)',
  MANUAL_EXIT: 'var(--color-neutral)',
};

let _container = null;
let _listEl = null;
let _entryCount = 0;

function _createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function _formatTs(ts) {
  if (!ts) return '--:--:--';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

function _getColor(type) {
  return TYPE_COLORS[type] || 'var(--text-secondary)';
}

function _buildDOM() {
  const root = _createEl('div', 'signal-log');

  const header = _createEl('div', 'signal-log__header', 'Signal Log');
  header.style.fontFamily = 'var(--font-sans)';
  header.style.color = 'var(--text-secondary)';
  header.style.fontSize = '0.8rem';
  header.style.marginBottom = 'var(--space-2)';
  root.appendChild(header);

  _listEl = _createEl('div', 'signal-log__list');
  _listEl.style.maxHeight = '320px';
  _listEl.style.overflowY = 'auto';
  _listEl.style.fontFamily = 'var(--font-mono)';
  _listEl.style.fontSize = '0.8rem';
  _listEl.style.lineHeight = '1.6';
  root.appendChild(_listEl);

  _container.appendChild(root);
}

export function init(containerEl) {
  _container = containerEl;
  _entryCount = 0;
  _buildDOM();
}

export function append(signalData) {
  if (!signalData) return;

  const entry = _createEl('div', 'signal-log__entry');
  entry.style.whiteSpace = 'nowrap';

  const color = _getColor(signalData.type);
  entry.style.color = color;

  // [HH:MM:SS] TYPE instrument direction price
  const ts = _createEl('span', 'signal-log__ts', `[${_formatTs(signalData.ts)}]`);
  ts.style.color = 'var(--text-tertiary)';
  ts.style.marginRight = 'var(--space-1)';

  const typeSpan = _createEl('span', 'signal-log__type', signalData.type || 'UNKNOWN');
  typeSpan.style.marginRight = 'var(--space-1)';

  const instrument = _createEl('span', 'signal-log__instrument', signalData.instrument || '');
  instrument.style.marginRight = 'var(--space-1)';

  const direction = _createEl('span', 'signal-log__direction',
    (signalData.direction || '').toUpperCase());
  direction.style.marginRight = 'var(--space-1)';

  const price = _createEl('span', 'signal-log__price',
    signalData.price != null ? signalData.price.toFixed(2) : '');

  entry.appendChild(ts);
  entry.appendChild(typeSpan);
  entry.appendChild(instrument);
  entry.appendChild(direction);
  entry.appendChild(price);

  _listEl.appendChild(entry);
  _entryCount++;

  // Trim oldest entries
  while (_entryCount > MAX_ENTRIES) {
    const first = _listEl.firstChild;
    if (first) {
      _listEl.removeChild(first);
      _entryCount--;
    } else {
      break;
    }
  }

  // Auto-scroll to bottom
  _listEl.scrollTop = _listEl.scrollHeight;
}

export function clear() {
  _listEl.textContent = '';
  _entryCount = 0;
}
