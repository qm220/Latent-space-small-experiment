"""Language-decoder layer indices for activation capture."""

from __future__ import annotations


def sample_language_layers(n_layers: int, n_sample: int = 8) -> list[int]:
    """Evenly spaced indices from ~1/8 depth through the last block.

    8B used 8/17/35 of 36 (early-mid, mid, last). Larger models sample more
    depths so PCA is not only three slices.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if n_sample < 1:
        raise ValueError(f"n_sample must be positive, got {n_sample}")
    if n_sample >= n_layers:
        return list(range(n_layers))
    lo = max(1, n_layers // 8)
    hi = n_layers - 1
    if n_sample == 1:
        return [hi]
    step = lo
    chosen = list(range(lo, hi, step))
    if not chosen or chosen[-1] != hi:
        chosen.append(hi)
    if len(chosen) == n_sample:
        return chosen
    picks = [round(lo + i * (hi - lo) / (n_sample - 1)) for i in range(n_sample)]
    ordered: list[int] = []
    for idx in picks:
        idx = min(max(idx, 0), hi)
        if not ordered or idx > ordered[-1]:
            ordered.append(idx)
    if ordered[-1] != hi:
        ordered[-1] = hi
    return ordered


def overview_layers(layers: list[int] | tuple[int, ...], n: int = 3) -> list[int]:
    """Subset for 3-panel overview plots: first, middle, last captured layer."""
    ordered = sorted(int(idx) for idx in layers)
    if len(ordered) <= n:
        return ordered
    if n == 1:
        return [ordered[-1]]
    picks = [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]
    seen: list[int] = []
    for idx in picks:
        if idx not in seen:
            seen.append(idx)
    return seen
