"""False-discovery-rate helpers shared by discovery and validation."""


def benjamini_hochberg(
    p_values: list[tuple[str, float]],
    alpha: float = 0.05,
    total_tests: int | None = None,
) -> dict[str, dict]:
    """Apply Benjamini-Hochberg FDR correction to a set of p-values."""
    valid = [(sid, p) for sid, p in p_values if p is not None]
    if not valid:
        return {}

    valid.sort(key=lambda x: x[1])
    m = total_tests if total_tests is not None else len(valid)
    if m < len(valid):
        raise ValueError(
            f"total_tests ({m}) < valid p-values ({len(valid)}): "
            "BH requires m >= n to maintain FDR control (Benjamini & Hochberg 1995)"
        )

    results = {}
    prev_adj = 1.0
    for rank_idx in range(len(valid) - 1, -1, -1):
        sid, raw_p = valid[rank_idx]
        rank = rank_idx + 1
        adj_p = min(raw_p * m / rank, 1.0)
        adj_p = min(adj_p, prev_adj)
        prev_adj = adj_p
        results[sid] = {
            "raw_p": raw_p,
            "adjusted_p": round(adj_p, 6),
            "fdr_significant": adj_p < alpha,
            "fdr_rank": rank,
        }

    return results
