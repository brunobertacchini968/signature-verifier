"""Tests básicos del módulo de preprocesamiento."""

from __future__ import annotations

import numpy as np

from src.preprocessing import (
    binarize_otsu,
    crop_to_bounding_box,
    denoise,
    preprocess,
    resize_with_aspect_ratio,
)


def _synthetic_signature(h: int = 200, w: int = 400) -> np.ndarray:
    """Genera una imagen sintética con un trazo oscuro sobre fondo claro."""
    img = np.full((h, w), 240, dtype=np.uint8)
    img[80:120, 50:350] = 30      # un trazo horizontal
    img[60:140, 180:220] = 30     # un trazo vertical que cruza
    return img


def test_binarize_otsu_inverts_correctly() -> None:
    img = _synthetic_signature()
    binary = binarize_otsu(img)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary).tolist()).issubset({0, 255})
    # La firma (originalmente oscura) debería quedar en 255.
    assert binary[100, 200] == 255
    assert binary[10, 10] == 0


def test_denoise_removes_isolated_pixels() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10, 10] = 255  # pixel aislado (ruido)
    binary[20:30, 20:30] = 255  # bloque grande (firma real)
    cleaned = denoise(binary, kernel_size=3)
    assert cleaned[10, 10] == 0
    assert cleaned[25, 25] == 255


def test_crop_to_bounding_box_reduces_size() -> None:
    binary = np.zeros((200, 200), dtype=np.uint8)
    binary[80:120, 90:110] = 255
    cropped = crop_to_bounding_box(binary, margin=0)
    assert cropped.shape[0] <= 40 + 2
    assert cropped.shape[1] <= 20 + 2


def test_resize_preserves_target_shape() -> None:
    img = np.full((100, 300), 255, dtype=np.uint8)
    out = resize_with_aspect_ratio(img, target_size=(512, 256))
    assert out.shape == (256, 512)
    assert out.dtype == np.uint8


def test_full_pipeline_returns_normalized_shape() -> None:
    img = _synthetic_signature()
    out = preprocess(img, target_size=(512, 256))
    assert out.shape == (256, 512)
    assert out.dtype == np.uint8
    assert (out > 0).sum() > 0  # debería haber píxeles activos
