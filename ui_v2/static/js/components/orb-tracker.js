/**
 * ORB Tracker — SVG progress ring + high/low/size display.
 *
 * Shows ORB aperture progress (e.g., 3/5 bars), current high/low/size,
 * and filter qualification badges.
 */

const RING_RADIUS = 40;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
const RING_STROKE = 5;
const SVG_SIZE = (RING_RADIUS + RING_STROKE) * 2 + 4;
const SVG_CENTER = SVG_SIZE / 2;

let _container = null;
let _progressCircle = null;
let _centerText = null;
let _highVal = null;
let _lowVal = null;
let _sizeVal = null;
let _badgesRow = null;
let _sessionLabel = null;

function _createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function _createSvgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

function _buildDOM() {
  const root = _createEl('div', 'orb-tracker');

  // Session label
  _sessionLabel = _createEl('div', 'orb-tracker__session', '--');
  _sessionLabel.style.fontFamily = 'var(--font-sans)';
  _sessionLabel.style.color = 'var(--text-secondary)';
  _sessionLabel.style.textAlign = 'center';
  _sessionLabel.style.marginBottom = 'var(--space-2)';
  root.appendChild(_sessionLabel);

  // SVG ring
  const svgWrap = _createEl('div', 'orb-tracker__ring');
  svgWrap.style.display = 'flex';
  svgWrap.style.justifyContent = 'center';

  const svg = _createSvgEl('svg', {
    width: SVG_SIZE, height: SVG_SIZE, viewBox: `0 0 ${SVG_SIZE} ${SVG_SIZE}`,
  });

  // Background circle
  svg.appendChild(_createSvgEl('circle', {
    cx: SVG_CENTER, cy: SVG_CENTER, r: RING_RADIUS,
    fill: 'none', stroke: 'var(--surface-border)', 'stroke-width': RING_STROKE,
  }));

  // Progress circle
  _progressCircle = _createSvgEl('circle', {
    cx: SVG_CENTER, cy: SVG_CENTER, r: RING_RADIUS,
    fill: 'none', stroke: 'var(--state-orb-forming)', 'stroke-width': RING_STROKE,
    'stroke-dasharray': RING_CIRCUMFERENCE,
    'stroke-dashoffset': RING_CIRCUMFERENCE,
    'stroke-linecap': 'round',
    transform: `rotate(-90 ${SVG_CENTER} ${SVG_CENTER})`,
  });
  svg.appendChild(_progressCircle);

  // Center text
  _centerText = _createSvgEl('text', {
    x: SVG_CENTER, y: SVG_CENTER,
    'text-anchor': 'middle', 'dominant-baseline': 'central',
    fill: 'var(--text-primary)', 'font-family': 'var(--font-mono)', 'font-size': '18',
  });
  _centerText.textContent = '0/0';
  svg.appendChild(_centerText);

  svgWrap.appendChild(svg);
  root.appendChild(svgWrap);

  // Stats row: HIGH / LOW / SIZE
  const stats = _createEl('div', 'orb-tracker__stats');
  stats.style.display = 'flex';
  stats.style.justifyContent = 'space-around';
  stats.style.marginTop = 'var(--space-3)';

  const makeStatCol = (label) => {
    const col = _createEl('div', 'orb-tracker__stat');
    col.style.textAlign = 'center';
    const lbl = _createEl('div', 'orb-tracker__stat-label', label);
    lbl.style.fontFamily = 'var(--font-sans)';
    lbl.style.color = 'var(--text-tertiary)';
    lbl.style.fontSize = '0.75rem';
    const val = _createEl('div', 'orb-tracker__stat-value mono', '--');
    val.style.color = 'var(--text-primary)';
    col.appendChild(lbl);
    col.appendChild(val);
    return { col, val };
  };

  const highStat = makeStatCol('HIGH');
  const lowStat = makeStatCol('LOW');
  const sizeStat = makeStatCol('SIZE');
  _highVal = highStat.val;
  _lowVal = lowStat.val;
  _sizeVal = sizeStat.val;
  stats.appendChild(highStat.col);
  stats.appendChild(lowStat.col);
  stats.appendChild(sizeStat.col);
  root.appendChild(stats);

  // Badges row
  _badgesRow = _createEl('div', 'orb-tracker__badges');
  _badgesRow.style.display = 'flex';
  _badgesRow.style.flexWrap = 'wrap';
  _badgesRow.style.gap = 'var(--space-1)';
  _badgesRow.style.marginTop = 'var(--space-2)';
  _badgesRow.style.justifyContent = 'center';
  root.appendChild(_badgesRow);

  _container.appendChild(root);
}

export function init(containerEl) {
  _container = containerEl;
  _buildDOM();
}

export function render(data) {
  if (!data) return;

  // Session label
  _sessionLabel.textContent = data.session || '--';

  // Ring progress
  const fraction = data.bars_total > 0 ? data.bars_elapsed / data.bars_total : 0;
  const offset = RING_CIRCUMFERENCE * (1 - Math.min(fraction, 1));
  _progressCircle.setAttribute('stroke-dashoffset', offset);

  // Update color: forming vs complete
  const strokeColor = fraction >= 1 ? 'var(--state-in-session)' : 'var(--state-orb-forming)';
  _progressCircle.setAttribute('stroke', strokeColor);

  // Center text
  _centerText.textContent = `${data.bars_elapsed}/${data.bars_total}`;

  // Stats
  _highVal.textContent = data.high != null ? data.high.toFixed(2) : '--';
  _lowVal.textContent = data.low != null ? data.low.toFixed(2) : '--';
  _sizeVal.textContent = data.size != null ? data.size.toFixed(2) : '--';

  // Badges
  _badgesRow.textContent = '';
  if (data.qualifications) {
    for (const [name, qualified] of Object.entries(data.qualifications)) {
      const badge = document.createElement('span');
      badge.className = 'badge orb-tracker__badge';
      badge.textContent = name;
      badge.style.backgroundColor = qualified ? 'var(--color-long)' : 'var(--color-short)';
      badge.style.color = '#000';
      badge.style.fontSize = '0.7rem';
      _badgesRow.appendChild(badge);
    }
  }
}
