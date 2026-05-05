"""Tests básicos de los features."""

from __future__ import annotations

import numpy as np

from src.features import (
    FEATURE_NAMES,
    compute_features,
    feature_anisotropic_scaling,
    feature_chain_code_distance,
    feature_glcm_distance,
    feature_hu_distance,
    feature_simple_geometric_distance,
    feature_ssim,
    feature_xor_ratio,
)


def _signature(h: int = 256, w: int = 512, shift: int = 0) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[120:140, 100 + shift : 400 + shift] = 255
    return img


def test_ssim_identical_is_one() -> None:
    a = _signature()
    assert feature_ssim(a, a) > 0.99


def test_xor_ratio_identical_is_one() -> None:
    a = _signature()
    assert feature_xor_ratio(a, a) == 1.0


def test_xor_ratio_disjoint_is_zero() -> None:
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((100, 100), dtype=np.uint8)
    a[10:20, 10:20] = 255
    b[50:60, 50:60] = 255
    assert feature_xor_ratio(a, b) == 0.0


def test_hu_distance_identical_is_zero() -> None:
    a = _signature()
    assert feature_hu_distance(a, a) < 1e-6


def test_anisotropic_scaling_identity_is_zero() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert feature_anisotropic_scaling(matrix) == 0.0


def test_anisotropic_scaling_isotropic_is_zero() -> None:
    matrix = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0]])
    assert feature_anisotropic_scaling(matrix) < 1e-6


def test_anisotropic_scaling_anisotropic_is_positive() -> None:
    matrix = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert feature_anisotropic_scaling(matrix) > 0.5


def test_glcm_distance_identical_is_small() -> None:
    a = _signature()
    assert feature_glcm_distance(a, a) < 1e-6


def test_chain_code_distance_identical_is_zero() -> None:
    a = _signature()
    assert feature_chain_code_distance(a, a) < 1e-6


def test_simple_geometric_identical_is_zero() -> None:
    a = _signature()
    assert feature_simple_geometric_distance(a, a) < 1e-6


def test_compute_features_returns_full_vector() -> None:
    a = _signature()
    b = _signature(shift=5)
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]])
    fv = compute_features(
        img_a_aligned=a,
        img_b=b,
        alignment_matrix=matrix,
        inlier_ratio=0.8,
    )
    assert fv.values.shape == (len(FEATURE_NAMES),)
    assert all(np.isfinite(v) for v in fv.values)
