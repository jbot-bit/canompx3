import pytest


class TestSlowCheckLabelsConsistency:
    """Guard the fast-skip registry against silent label drift."""

    def test_current_slow_labels_are_all_in_checks(self):
        from pipeline import check_drift

        known_labels = {label for label, *_ in check_drift.CHECKS}
        stale = check_drift.SLOW_CHECK_LABELS - known_labels
        assert stale == set(), (
            f"SLOW_CHECK_LABELS contains label(s) not in CHECKS: {sorted(stale)}. "
            "Either a check was renamed/removed without updating SLOW_CHECK_LABELS, "
            "or there is a typo. Re-run scripts/tools/profile_check_drift.py."
        )

    def test_assert_passes_on_clean_state(self):
        from pipeline import check_drift

        check_drift._assert_slow_labels_valid()

    def test_assert_raises_on_stale_label(self, monkeypatch):
        from pipeline import check_drift

        poisoned = frozenset(check_drift.SLOW_CHECK_LABELS | {"Definitely not a real check label xyzzy"})
        monkeypatch.setattr(check_drift, "SLOW_CHECK_LABELS", poisoned)

        with pytest.raises(RuntimeError) as excinfo:
            check_drift._assert_slow_labels_valid()

        msg = str(excinfo.value)
        assert "Definitely not a real check label xyzzy" in msg
        assert "SLOW_CHECK_LABELS" in msg

    def test_assert_raises_on_all_stale(self, monkeypatch):
        from pipeline import check_drift

        monkeypatch.setattr(check_drift, "CHECKS", [])

        with pytest.raises(RuntimeError) as excinfo:
            check_drift._assert_slow_labels_valid()

        msg = str(excinfo.value)
        sample = next(iter(check_drift.SLOW_CHECK_LABELS))
        assert sample in msg
