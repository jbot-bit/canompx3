"""Meta-labeling ML module for ORB breakout signal filtering.

Architecture:
  Primary model = ORB rule-based system (provides BUY/SELL signals)
  Meta-label    = RF classifier (provides P(win) → skip/take decision)

Per de Prado (AIFML): "The decision regarding the direction of a trade
(the sign) and the size of the position are made by independent algorithms."
"""

__version__ = "2.0.0"
