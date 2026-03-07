/**
 * Trade Status — live trade monitoring panel.
 *
 * Shows entry/stop/target levels, direction, live R, time-in-trade.
 */

let _container = null;
let _root = null;
let _headerInstrument = null;
let _headerBadge = null;
let _entryVal = null;
let _stopVal = null;
let _targetVal = null;
let _liveR = null;
let _timeVal = null;
let _sessionLabel = null;

function _createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function _makePriceRow(label) {
  const row = _createEl('div', 'trade-status__row');
  row.style.display = 'flex';
  row.style.justifyContent = 'space-between';
  row.style.padding = 'var(--space-1) 0';

  const lbl = _createEl('span', 'trade-status__label', label);
  lbl.style.fontFamily = 'var(--font-sans)';
  lbl.style.color = 'var(--text-secondary)';

  const val = _createEl('span', 'trade-status__value mono', '--');
  val.style.color = 'var(--text-primary)';

  row.appendChild(lbl);
  row.appendChild(val);
  return { row, val };
}

function _buildDOM() {
  _root = _createEl('div', 'trade-status card');

  // Header
  const header = _createEl('div', 'trade-status__header');
  header.style.display = 'flex';
  header.style.alignItems = 'center';
  header.style.gap = 'var(--space-2)';
  header.style.marginBottom = 'var(--space-3)';

  _headerInstrument = _createEl('span', 'trade-status__instrument mono', '--');
  _headerInstrument.style.fontSize = '1.1rem';
  _headerInstrument.style.fontWeight = '600';

  _headerBadge = _createEl('span', 'badge trade-status__direction', '--');
  _headerBadge.style.color = '#000';

  header.appendChild(_headerInstrument);
  header.appendChild(_headerBadge);
  _root.appendChild(header);

  // Price levels
  const entry = _makePriceRow('Entry');
  const stop = _makePriceRow('Stop');
  const target = _makePriceRow('Target');
  _entryVal = entry.val;
  _stopVal = stop.val;
  _targetVal = target.val;

  _stopVal.style.color = 'var(--color-short)';
  _targetVal.style.color = 'var(--color-long)';

  _root.appendChild(entry.row);
  _root.appendChild(stop.row);
  _root.appendChild(target.row);

  // Separator
  const sep = _createEl('hr', 'trade-status__sep');
  sep.style.border = 'none';
  sep.style.borderTop = '1px solid var(--surface-border)';
  sep.style.margin = 'var(--space-2) 0';
  _root.appendChild(sep);

  // Live R
  const rRow = _createEl('div', 'trade-status__r-row');
  rRow.style.textAlign = 'center';
  rRow.style.margin = 'var(--space-2) 0';

  const rLabel = _createEl('div', 'trade-status__r-label', 'Live R');
  rLabel.style.fontFamily = 'var(--font-sans)';
  rLabel.style.color = 'var(--text-tertiary)';
  rLabel.style.fontSize = '0.75rem';

  _liveR = _createEl('div', 'trade-status__r-value mono', '--');
  _liveR.style.fontSize = '1.8rem';
  _liveR.style.fontWeight = '700';

  rRow.appendChild(rLabel);
  rRow.appendChild(_liveR);
  _root.appendChild(rRow);

  // Time in trade
  _timeVal = _createEl('div', 'trade-status__time mono', '--');
  _timeVal.style.textAlign = 'center';
  _timeVal.style.color = 'var(--text-secondary)';
  _root.appendChild(_timeVal);

  // Session
  _sessionLabel = _createEl('div', 'trade-status__session', '--');
  _sessionLabel.style.textAlign = 'center';
  _sessionLabel.style.fontFamily = 'var(--font-sans)';
  _sessionLabel.style.color = 'var(--text-tertiary)';
  _sessionLabel.style.fontSize = '0.75rem';
  _sessionLabel.style.marginTop = 'var(--space-1)';
  _root.appendChild(_sessionLabel);

  _container.appendChild(_root);
}

function _formatTime(minutes) {
  if (minutes == null) return '--';
  const m = Math.floor(minutes);
  const s = Math.round((minutes - m) * 60);
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

export function init(containerEl) {
  _container = containerEl;
  _buildDOM();
}

export function render(data) {
  if (!data) return;

  _headerInstrument.textContent = data.instrument || '--';

  const dir = (data.direction || '').toUpperCase();
  _headerBadge.textContent = dir || '--';
  _headerBadge.style.backgroundColor = dir === 'LONG'
    ? 'var(--color-long)' : dir === 'SHORT'
      ? 'var(--color-short)' : 'var(--text-tertiary)';

  _entryVal.textContent = data.entry_price != null ? data.entry_price.toFixed(2) : '--';
  _stopVal.textContent = data.stop_price != null ? data.stop_price.toFixed(2) : '--';
  _targetVal.textContent = data.target_price != null ? data.target_price.toFixed(2) : '--';

  const r = data.unrealized_r;
  if (r != null) {
    const sign = r >= 0 ? '+' : '';
    _liveR.textContent = `${sign}${r.toFixed(2)}R`;
    _liveR.style.color = r >= 0 ? 'var(--color-long)' : 'var(--color-short)';
  } else {
    _liveR.textContent = '--';
    _liveR.style.color = 'var(--text-secondary)';
  }

  _timeVal.textContent = _formatTime(data.time_in_trade_minutes);
  _sessionLabel.textContent = data.session || '--';
}

export function clear() {
  _headerInstrument.textContent = '--';
  _headerBadge.textContent = '--';
  _headerBadge.style.backgroundColor = 'var(--text-tertiary)';
  _entryVal.textContent = '--';
  _stopVal.textContent = '--';
  _targetVal.textContent = '--';
  _liveR.textContent = '--';
  _liveR.style.color = 'var(--text-secondary)';
  _timeVal.textContent = '--';
  _sessionLabel.textContent = '--';
}
