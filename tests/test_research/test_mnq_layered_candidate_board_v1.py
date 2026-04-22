from research.mnq_layered_candidate_board_v1 import (
    BASE_FEATURES,
    EXCLUDED_PAIRS,
    PARENTS,
    signal_specs,
)


def test_signal_specs_match_prereg_count():
    specs = signal_specs()
    assert len(BASE_FEATURES) == 9
    assert len(EXCLUDED_PAIRS) == 5
    assert len(specs) == 40
    assert len(PARENTS) == 6
    assert len(specs) * len(PARENTS) == 240


def test_signal_specs_have_unique_names():
    names = [name for name, _ in signal_specs()]
    assert len(names) == len(set(names))


def test_excluded_pairs_not_emitted():
    emitted = {tuple(sorted(inputs)) for _, inputs in signal_specs() if len(inputs) == 2}
    for pair in EXCLUDED_PAIRS:
        assert pair not in emitted
