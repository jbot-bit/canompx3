/**
 * Manual Trade — Signal-Only mode entry/exit form.
 *
 * For traders not using the automated orchestrator.
 * Toggleable with M key.
 */

const INSTRUMENTS = ['MGC', 'MNQ', 'MES', 'M2K'];
const DIRECTIONS = ['Long', 'Short'];

let _container = null;
let _root = null;
let _visible = false;
let _mode = 'entry'; // 'entry' | 'exit'

// Entry fields
let _instrumentSelect = null;
let _directionSelect = null;
let _priceInput = null;
let _orbHighInput = null;
let _orbLowInput = null;

// Exit fields
let _exitPriceInput = null;

// Panels
let _entryPanel = null;
let _exitPanel = null;
let _entryModeBtn = null;
let _exitModeBtn = null;

function _createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function _createInput(placeholder) {
  const input = _createEl('input', 'manual-trade__input');
  input.type = 'number';
  input.step = 'any';
  input.placeholder = placeholder;
  input.style.fontFamily = 'var(--font-mono)';
  return input;
}

function _createSelect(options) {
  const sel = document.createElement('select');
  sel.className = 'manual-trade__select';
  for (const opt of options) {
    const o = _createEl('option', null, opt);
    o.value = opt;
    sel.appendChild(o);
  }
  return sel;
}

function _createLabel(text) {
  const lbl = _createEl('label', 'manual-trade__label', text);
  lbl.style.fontFamily = 'var(--font-sans)';
  lbl.style.color = 'var(--text-secondary)';
  lbl.style.fontSize = '0.75rem';
  lbl.style.display = 'block';
  lbl.style.marginTop = 'var(--space-2)';
  lbl.style.marginBottom = 'var(--space-1)';
  return lbl;
}

function _setMode(mode) {
  _mode = mode;
  _entryPanel.style.display = mode === 'entry' ? 'block' : 'none';
  _exitPanel.style.display = mode === 'exit' ? 'block' : 'none';
  _entryModeBtn.className = mode === 'entry' ? 'btn btn--primary' : 'btn btn--ghost';
  _exitModeBtn.className = mode === 'exit' ? 'btn btn--primary' : 'btn btn--ghost';
}

async function _submitEntry() {
  const payload = {
    action: 'entry',
    instrument: _instrumentSelect.value,
    direction: _directionSelect.value.toLowerCase(),
    price: parseFloat(_priceInput.value),
    orb_high: parseFloat(_orbHighInput.value),
    orb_low: parseFloat(_orbLowInput.value),
  };
  try {
    await fetch('/api/trade-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.error('Manual entry POST failed:', e);
  }
}

async function _submitExit() {
  const payload = {
    action: 'exit',
    price: parseFloat(_exitPriceInput.value),
  };
  try {
    await fetch('/api/trade-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.error('Manual exit POST failed:', e);
  }
}

function _buildDOM() {
  _root = _createEl('div', 'manual-trade card');
  _root.style.display = 'none';

  // Header
  const header = _createEl('div', 'manual-trade__header', 'Manual Trade');
  header.style.fontFamily = 'var(--font-sans)';
  header.style.color = 'var(--text-secondary)';
  header.style.fontSize = '0.85rem';
  header.style.marginBottom = 'var(--space-3)';
  _root.appendChild(header);

  // Mode toggle
  const modeRow = _createEl('div', 'manual-trade__mode');
  modeRow.style.display = 'flex';
  modeRow.style.gap = 'var(--space-2)';
  modeRow.style.marginBottom = 'var(--space-3)';

  _entryModeBtn = _createEl('button', 'btn btn--primary', 'Entry');
  _exitModeBtn = _createEl('button', 'btn btn--ghost', 'Exit');
  _entryModeBtn.addEventListener('click', () => _setMode('entry'));
  _exitModeBtn.addEventListener('click', () => _setMode('exit'));
  modeRow.appendChild(_entryModeBtn);
  modeRow.appendChild(_exitModeBtn);
  _root.appendChild(modeRow);

  // Entry panel
  _entryPanel = _createEl('div', 'manual-trade__entry-panel');

  _instrumentSelect = _createSelect(INSTRUMENTS);
  _directionSelect = _createSelect(DIRECTIONS);
  _priceInput = _createInput('Entry price');
  _orbHighInput = _createInput('ORB high');
  _orbLowInput = _createInput('ORB low');

  _entryPanel.appendChild(_createLabel('Instrument'));
  _entryPanel.appendChild(_instrumentSelect);
  _entryPanel.appendChild(_createLabel('Direction'));
  _entryPanel.appendChild(_directionSelect);
  _entryPanel.appendChild(_createLabel('Price'));
  _entryPanel.appendChild(_priceInput);
  _entryPanel.appendChild(_createLabel('ORB High'));
  _entryPanel.appendChild(_orbHighInput);
  _entryPanel.appendChild(_createLabel('ORB Low'));
  _entryPanel.appendChild(_orbLowInput);

  const logEntryBtn = _createEl('button', 'btn btn--success manual-trade__submit', 'Log Entry');
  logEntryBtn.style.marginTop = 'var(--space-3)';
  logEntryBtn.style.width = '100%';
  logEntryBtn.addEventListener('click', _submitEntry);
  _entryPanel.appendChild(logEntryBtn);

  _root.appendChild(_entryPanel);

  // Exit panel
  _exitPanel = _createEl('div', 'manual-trade__exit-panel');
  _exitPanel.style.display = 'none';

  _exitPriceInput = _createInput('Exit price');

  _exitPanel.appendChild(_createLabel('Price'));
  _exitPanel.appendChild(_exitPriceInput);

  const logExitBtn = _createEl('button', 'btn btn--danger manual-trade__submit', 'Log Exit');
  logExitBtn.style.marginTop = 'var(--space-3)';
  logExitBtn.style.width = '100%';
  logExitBtn.addEventListener('click', _submitExit);
  _exitPanel.appendChild(logExitBtn);

  _root.appendChild(_exitPanel);
  _container.appendChild(_root);
}

export function init(containerEl) {
  _container = containerEl;
  _buildDOM();
}

export function show() {
  _visible = true;
  _root.style.display = 'block';
}

export function hide() {
  _visible = false;
  _root.style.display = 'none';
}

export function toggle() {
  if (_visible) hide();
  else show();
}

export function isVisible() {
  return _visible;
}
