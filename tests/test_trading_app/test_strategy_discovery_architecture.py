from pathlib import Path

from trading_app.fdr import benjamini_hochberg
from trading_app.strategy_validator import benjamini_hochberg as validator_bh


def test_strategy_discovery_does_not_import_validator_for_fdr_helper():
    source = Path("trading_app/strategy_discovery.py").read_text(encoding="utf-8")

    assert "from trading_app.strategy_validator import benjamini_hochberg" not in source
    assert "from trading_app.fdr import benjamini_hochberg" in source


def test_neutral_bh_helper_matches_expected_adjustment():
    result = benjamini_hochberg([("a", 0.01), ("b", 0.04), ("c", 0.20)], alpha=0.05)

    assert result["a"]["fdr_significant"] is True
    assert result["a"]["adjusted_p"] == 0.03
    assert result["b"]["fdr_significant"] is False


def test_neutral_bh_helper_matches_validator_compatibility_export():
    p_values = [("a", 0.01), ("b", 0.04), ("c", 0.20)]

    assert benjamini_hochberg(p_values, alpha=0.05) == validator_bh(p_values, alpha=0.05)
