"""Perturbaciones controladas para análisis de robustez.

Cada función toma una imagen (uint8, grayscale o binaria) y un parámetro
de intensidad, y devuelve la imagen perturbada. Pensadas para aplicarse
ANTES del preprocesamiento, simulando degradaciones reales.
"""

from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# ROTACIÓN
# ============================================================

def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota la imagen alrededor de su centro. Borde negro (fondo)."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # fondo blanco para preservar binarización Otsu posterior
    )


# ============================================================
# RUIDO
# ============================================================

def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Ruido gaussiano aditivo con desvío sigma."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper(img: np.ndarray, density: float) -> np.ndarray:
    """Ruido sal y pimienta: density es la fracción total de píxeles afectados."""
    out = img.copy()
    n = int(density * img.size / 2)
    if n <= 0:
        return out
    rng = np.random.default_rng()
    # Sal (255)
    coords_s = (rng.integers(0, img.shape[0], n), rng.integers(0, img.shape[1], n))
    out[coords_s] = 255
    # Pimienta (0)
    coords_p = (rng.integers(0, img.shape[0], n), rng.integers(0, img.shape[1], n))
    out[coords_p] = 0
    return out


# ============================================================
# ESCALA
# ============================================================

def rescale(img: np.ndarray, factor: float) -> np.ndarray:
    """Reescala la imagen y la pega centrada en un canvas del tamaño original."""
    h, w = img.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)

    canvas = np.full((h, w), 255, dtype=np.uint8)  # fondo blanco
    if new_w <= w and new_h <= h:
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    else:
        # Si crece más que el canvas, recortar centrado.
        x_off = (new_w - w) // 2
        y_off = (new_h - h) // 2
        canvas = resized[y_off:y_off + h, x_off:x_off + w]
    return canvas


# ============================================================
# OCLUSIÓN
# ============================================================

def occlude(img: np.ndarray, fraction: float) -> np.ndarray:
    """Tapa un parche rectangular de área 'fraction' (0-1) del bbox de la firma."""
    out = img.copy()
    h, w = img.shape[:2]
    # Buscar el bbox del contenido para que la oclusión sea relevante.
    inv = 255 - out  # asumimos firma oscura sobre fondo claro
    coords = cv2.findNonZero((inv > 50).astype(np.uint8))
    if coords is None:
        return out
    x, y, bw, bh = cv2.boundingRect(coords)

    occ_w = max(1, int(np.sqrt(fraction) * bw))
    occ_h = max(1, int(np.sqrt(fraction) * bh))

    rng = np.random.default_rng()
    ox = x + rng.integers(0, max(1, bw - occ_w + 1))
    oy = y + rng.integers(0, max(1, bh - occ_h + 1))

    out[oy:oy + occ_h, ox:ox + occ_w] = 255  # cuadrado blanco (fondo)
    return out


# ============================================================
# CATÁLOGO DE PERTURBACIONES PARA EXPERIMENTOS
# ============================================================

PERTURBATION_GRID: dict[str, list[tuple[str, callable, dict]]] = {
    "rotation": [
        ("rot_5",  rotate, {"angle_deg": 5}),
        ("rot_15", rotate, {"angle_deg": 15}),
        ("rot_30", rotate, {"angle_deg": 30}),
        ("rot_45", rotate, {"angle_deg": 45}),
        ("rot_90", rotate, {"angle_deg": 90}),
    ],
    "gaussian_noise": [
        ("gauss_5",  add_gaussian_noise, {"sigma": 5}),
        ("gauss_10", add_gaussian_noise, {"sigma": 10}),
        ("gauss_20", add_gaussian_noise, {"sigma": 20}),
    ],
    "salt_pepper": [
        ("sp_1",  add_salt_and_pepper, {"density": 0.01}),
        ("sp_5",  add_salt_and_pepper, {"density": 0.05}),
        ("sp_10", add_salt_and_pepper, {"density": 0.10}),
    ],
    "scale": [
        ("scale_0.5",  rescale, {"factor": 0.5}),
        ("scale_0.75", rescale, {"factor": 0.75}),
        ("scale_1.5",  rescale, {"factor": 1.5}),
        ("scale_2.0",  rescale, {"factor": 2.0}),
    ],
    "occlusion": [
        ("occ_10", occlude, {"fraction": 0.10}),
        ("occ_25", occlude, {"fraction": 0.25}),
        ("occ_40", occlude, {"fraction": 0.40}),
    ],
}


def apply_perturbation(
    img: np.ndarray, fn: callable, **kwargs
) -> np.ndarray:
    """Wrapper para aplicar una perturbación con sus parámetros."""
    return fn(img, **kwargs)
