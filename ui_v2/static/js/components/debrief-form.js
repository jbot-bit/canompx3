/**
 * Debrief Form — fast-path clean trade + expandable details form.
 *
 * Space → submit "Clean Trade" (FOLLOWED_PLAN, no notes)
 * D → toggle expanded form visibility
 */

const ADHERENCE_OPTIONS = [
  'FOLLOWED_PLAN',
  'MINOR_DEVIATION',
  'MAJOR_DEVIATION',
  'REVENGE_TRADE',
  'FOMO_ENTRY',
];

let _container = null;
let _tradeData = null;
let _visible = false;
let _detailsExpanded = false;

// DOM refs
let _root = null;
let _summaryEl = null;
let _detailsPanel = null;
let _adherenceSelect = null;
let _deviationInput = null;
let _notesTextarea = null;
let _letterTextarea = null;

function _buildSummaryText(d) {
  const sign = d.pnl_r >= 0 ? '+' : '';
  const dir = d.direction.toUpperCase();
  const dur = d.time_held_minutes + 'm';
  return `${d.instrument} ${dir} ${sign}${d.pnl_r.toFixed(1)}R | ${dur}`;
}

function _createEl(tag, className, textContent) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (textContent !== undefined) el.textContent = textContent;
  return el;
}

function _buildDOM() {
  _root = _createEl('div', 'debrief');
  _root.style.display = 'none';

  // Summary card
  _summaryEl = _createEl('div', 'debrief__summary card');
  _root.appendChild(_summaryEl);

  // Button row
  const btnRow = _createEl('div', 'debrief__actions');
  btnRow.style.display = 'flex';
  btnRow.style.gap = 'var(--space-3)';
  btnRow.style.marginTop = 'var(--space-3)';

  const cleanBtn = _createEl('button', 'btn btn--success debrief__clean-btn');
  const kbdSpace = _createEl('span', 'kbd', 'Space');
  cleanBtn.appendChild(kbdSpace);
  cleanBtn.appendChild(document.createTextNode(' Clean Trade'));
  cleanBtn.addEventListener('click', submitCleanTrade);

  const detailsBtn = _createEl('button', 'btn btn--ghost debrief__details-btn');
  const kbdD = _createEl('span', 'kbd', 'D');
  detailsBtn.appendChild(kbdD);
  detailsBtn.appendChild(document.createTextNode(' Details\u2026'));
  detailsBtn.addEventListener('click', toggleDetails);

  btnRow.appendChild(cleanBtn);
  btnRow.appendChild(detailsBtn);
  _root.appendChild(btnRow);

  // Expanded details panel
  _detailsPanel = _createEl('div', 'debrief__details card');
  _detailsPanel.style.display = 'none';
  _detailsPanel.style.marginTop = 'var(--space-3)';

  // Adherence dropdown
  const adhLabel = _createEl('label', 'debrief__label', 'Adherence');
  adhLabel.style.fontFamily = 'var(--font-sans)';
  _adherenceSelect = document.createElement('select');
  _adherenceSelect.className = 'debrief__select';
  for (const opt of ADHERENCE_OPTIONS) {
    const o = _createEl('option', null, opt);
    o.value = opt;
    _adherenceSelect.appendChild(o);
  }

  // Deviation trigger
  const devLabel = _createEl('label', 'debrief__label', 'Deviation Trigger');
  _deviationInput = _createEl('input', 'debrief__input');
  _deviationInput.type = 'text';
  _deviationInput.placeholder = 'What triggered the deviation?';

  // Notes
  const notesLabel = _createEl('label', 'debrief__label', 'Notes');
  _notesTextarea = _createEl('textarea', 'debrief__textarea');
  _notesTextarea.rows = 3;
  _notesTextarea.placeholder = 'Trade notes...';

  // Letter
  const letterLabel = _createEl('label', 'debrief__label', 'Letter to Future Self');
  _letterTextarea = _createEl('textarea', 'debrief__textarea');
  _letterTextarea.rows = 3;
  _letterTextarea.placeholder = 'What should future-you remember?';

  // Submit
  const submitBtn = _createEl('button', 'btn btn--primary debrief__submit-btn', 'Submit Debrief');
  submitBtn.addEventListener('click', _submitDetailed);

  for (const el of [adhLabel, _adherenceSelect, devLabel, _deviationInput,
    notesLabel, _notesTextarea, letterLabel, _letterTextarea, submitBtn]) {
    _detailsPanel.appendChild(el);
  }
  _root.appendChild(_detailsPanel);
  _container.appendChild(_root);
}

function _renderSummary() {
  _summaryEl.textContent = '';
  if (!_tradeData) return;

  const line = _createEl('span', 'debrief__summary-text mono', _buildSummaryText(_tradeData));
  const badge = _createEl('span', 'badge debrief__direction-badge',
    _tradeData.direction.toUpperCase());
  badge.style.backgroundColor = _tradeData.direction.toUpperCase() === 'LONG'
    ? 'var(--color-long)' : 'var(--color-short)';
  badge.style.color = '#000';
  badge.style.marginLeft = 'var(--space-2)';

  _summaryEl.appendChild(line);
  _summaryEl.appendChild(badge);
}

async function _postDebrief(payload) {
  try {
    await fetch('/api/debrief', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.error('Debrief POST failed:', e);
  }
}

async function submitCleanTrade() {
  if (!_tradeData) return;
  await _postDebrief({
    strategy_id: _tradeData.strategy_id,
    adherence: 'FOLLOWED_PLAN',
    deviation_trigger: '',
    notes: '',
    letter: '',
  });
  hide();
}

async function _submitDetailed() {
  if (!_tradeData) return;
  await _postDebrief({
    strategy_id: _tradeData.strategy_id,
    adherence: _adherenceSelect.value,
    deviation_trigger: _deviationInput.value,
    notes: _notesTextarea.value,
    letter: _letterTextarea.value,
  });
  hide();
}

function toggleDetails() {
  _detailsExpanded = !_detailsExpanded;
  _detailsPanel.style.display = _detailsExpanded ? 'block' : 'none';
}

export function init(containerEl) {
  _container = containerEl;
  _buildDOM();
}

export function show(tradeData) {
  _tradeData = tradeData;
  _visible = true;
  _detailsExpanded = false;
  _detailsPanel.style.display = 'none';
  _adherenceSelect.value = 'FOLLOWED_PLAN';
  _deviationInput.value = '';
  _notesTextarea.value = '';
  _letterTextarea.value = '';
  _renderSummary();
  _root.style.display = 'block';
}

export function hide() {
  _visible = false;
  _root.style.display = 'none';
  _tradeData = null;
}

export function isVisible() {
  return _visible;
}

export { submitCleanTrade, toggleDetails };
