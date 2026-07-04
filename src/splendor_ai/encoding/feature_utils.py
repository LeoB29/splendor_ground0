"""Small helpers for versioned feature-vector compatibility."""

from __future__ import annotations

from collections.abc import Sequence


def fit_feature_vector(vector: Sequence[float], target_size: int) -> tuple[float, ...]:
    """Return ``vector`` truncated or zero-padded to ``target_size``."""

    if target_size < 0:
        raise ValueError("target_size must be non-negative.")
    fitted = tuple(float(value) for value in vector[:target_size])
    if len(fitted) == target_size:
        return fitted
    return fitted + (0.0,) * (target_size - len(fitted))
