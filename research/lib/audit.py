"""Join verification -- catches silent row inflation from bad JOINs."""


def assert_no_inflation(n_before: int, n_after: int, context: str = "") -> None:
    """Raise ValueError if a JOIN inflated row count.

    Usage:
        n_raw = len(outcomes_df)
        merged = outcomes_df.merge(features_df, on=[...])
        assert_no_inflation(n_raw, len(merged), context="my_analysis")
    """
    if n_after > n_before:
        tag = f" [{context}]" if context else ""
        raise ValueError(
            f"Row count inflated{tag}: {n_before} -> {n_after}. "
            f"Check JOIN columns (missing orb_minutes?)."
        )
