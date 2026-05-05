"""Tests del módulo evaluator (FAR/FRR/EER/ROC)."""

from __future__ import annotations

import numpy as np

from src.evaluator import evaluate


def test_perfect_separation_gives_zero_eer() -> None:
    sg = np.array([0.9, 0.95, 0.99])
    sf = np.array([0.1, 0.05, 0.01])
    metrics = evaluate(sg, sf)
    assert metrics.eer < 0.01
    assert metrics.auc > 0.99


def test_complete_overlap_gives_high_eer() -> None:
    rng = np.random.default_rng(0)
    sg = rng.uniform(0.4, 0.6, size=200)
    sf = rng.uniform(0.4, 0.6, size=200)
    metrics = evaluate(sg, sf)
    # Si las distribuciones se solapan completamente, el EER ronda 0.5.
    assert 0.3 < metrics.eer < 0.6
    assert 0.3 < metrics.auc < 0.7


def test_far_and_frr_are_monotonic() -> None:
    rng = np.random.default_rng(1)
    sg = rng.normal(0.7, 0.1, size=100)
    sf = rng.normal(0.3, 0.1, size=100)
    metrics = evaluate(sg, sf)
    # FAR decrece con el umbral (umbrales más altos rechazan más falsificaciones).
    assert metrics.far[0] >= metrics.far[-1]
    # FRR crece con el umbral.
    assert metrics.frr[0] <= metrics.frr[-1]
