"""Small, reproducible helpers for cross-city graph alignment experiments."""
from __future__ import annotations

import numpy as np


def spectral_graph_alignment(source_adjacency: np.ndarray, target_adjacency: np.ndarray,
                             dimensions: int = 8) -> np.ndarray:
    """Return a soft node mapping based on normalized Laplacian spectra."""
    def embedding(adjacency):
        adjacency = np.asarray(adjacency, dtype=float)
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency
        values, vectors = np.linalg.eigh(laplacian)
        order = np.argsort(values)[: min(dimensions, len(values))]
        return vectors[:, order]
    source = embedding(source_adjacency)
    target = embedding(target_adjacency)
    scores = np.exp(-((source[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1))
    return scores / (scores.sum(axis=1, keepdims=True) + 1e-12)


def few_shot_checkpoints(total_steps: int, steps_per_day: int = 288) -> dict[str, tuple[int, int]]:
    """Return 3-, 7-, and 14-day slices for target-city adaptation."""
    return {
        "3_days": (0, min(total_steps, 3 * steps_per_day)),
        "7_days": (0, min(total_steps, 7 * steps_per_day)),
        "14_days": (0, min(total_steps, 14 * steps_per_day)),
    }