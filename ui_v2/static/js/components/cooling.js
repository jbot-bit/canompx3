/**
 * Cooling — breathing animation overlay with letter from past self.
 *
 * Shows a breathing animation (CSS .breathing-circle already defined),
 * countdown timer, trader's letter from past self, and soft override button.
 */

let _overlayEl = null;
let _root = null;
let _countdownEl = null;
let _letterCard = null;
let _letterText = null;
let _active = false;

function _createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function _formatCountdown(seconds) {
  if (seconds == null || seconds < 0) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

async function _handleOverride() {
  try {
    const resp = await fetch('/api/cooling/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!resp.ok) {
      console.error(`Cooling override failed: ${resp.status}`);
    }
  } catch (e) {
    console.error('Cooling override POST failed:', e);
  }
}

function _buildDOM() {
  _root = _createEl('div', 'cooling');
  _root.style.display = 'none';
  _root.style.position = 'fixed';
  _root.style.inset = '0';
  _root.style.backgroundColor = 'rgba(10, 14, 20, 0.85)';
  _root.style.zIndex = '1000';
  _root.style.justifyContent = 'center';
  _root.style.alignItems = 'center';
  _root.style.flexDirection = 'column';

  // Centered content wrapper
  const content = _createEl('div', 'cooling__content');
  content.style.display = 'flex';
  content.style.flexDirection = 'column';
  content.style.alignItems = 'center';
  content.style.gap = 'var(--space-4)';
  content.style.maxWidth = '400px';
  content.style.textAlign = 'center';

  // Breathing circle
  const circle = _createEl('div', 'breathing-circle cooling__circle');
  content.appendChild(circle);

  // "Take a breath" text
  const breathText = _createEl('div', 'cooling__text', 'Take a breath');
  breathText.style.fontFamily = 'var(--font-sans)';
  breathText.style.color = 'var(--text-primary)';
  breathText.style.fontSize = '1.4rem';
  breathText.style.fontWeight = '300';
  content.appendChild(breathText);

  // Countdown
  _countdownEl = _createEl('div', 'cooling__countdown mono', '00:00');
  _countdownEl.style.fontSize = '2rem';
  _countdownEl.style.color = 'var(--text-secondary)';
  content.appendChild(_countdownEl);

  // Letter card (hidden by default)
  _letterCard = _createEl('div', 'cooling__letter card');
  _letterCard.style.display = 'none';
  _letterCard.style.padding = 'var(--space-4)';
  _letterCard.style.maxWidth = '360px';
  _letterCard.style.width = '100%';

  const letterHeader = _createEl('div', 'cooling__letter-header', 'Letter from past you');
  letterHeader.style.fontFamily = 'var(--font-sans)';
  letterHeader.style.color = 'var(--text-tertiary)';
  letterHeader.style.fontSize = '0.75rem';
  letterHeader.style.marginBottom = 'var(--space-2)';

  _letterText = _createEl('div', 'cooling__letter-text');
  _letterText.style.fontFamily = 'var(--font-sans)';
  _letterText.style.color = 'var(--text-primary)';
  _letterText.style.fontStyle = 'italic';
  _letterText.style.lineHeight = '1.6';

  _letterCard.appendChild(letterHeader);
  _letterCard.appendChild(_letterText);
  content.appendChild(_letterCard);

  // Override button
  const overrideBtn = _createEl('button', 'btn btn--ghost cooling__override',
    "I'm ready to continue");
  overrideBtn.style.marginTop = 'var(--space-4)';
  overrideBtn.style.fontSize = '0.85rem';
  overrideBtn.addEventListener('click', _handleOverride);
  content.appendChild(overrideBtn);

  _root.appendChild(content);
  _overlayEl.appendChild(_root);
}

export function init(overlayEl) {
  _overlayEl = overlayEl;
  _buildDOM();
}

export function render(data) {
  if (!data) return;

  if (data.active) {
    _active = true;
    _root.style.display = 'flex';

    // Countdown
    _countdownEl.textContent = _formatCountdown(data.remaining_seconds);

    // Letter
    if (data.letter_text) {
      _letterText.textContent = data.letter_text;
      _letterCard.style.display = 'block';
    } else {
      _letterCard.style.display = 'none';
    }
  } else {
    _active = false;
    _root.style.display = 'none';
  }
}

export function isActive() {
  return _active;
}
