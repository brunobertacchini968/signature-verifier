"""Preprocesamiento de imágenes de firma.

Pipeline:
    grayscale -> Otsu binarization -> morphological opening (denoise)
    -> bounding box crop -> padding a tamaño fijo conservando aspect ratio.

La salida es una imagen binaria (uint8, 0/255) con la firma en blanco
sobre fondo negro, lista para alineación geométrica.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_TARGET_SIZE = (512, 256)  # (width, height)


def load_image(path: str | Path) -> np.ndarray:
    """Carga una imagen desde disco en escala de grises."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    return img


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Binarización por Otsu. Devuelve firma en blanco (255), fondo en negro (0)."""
    # THRESH_BINARY_INV porque las firmas son tinta oscura sobre fondo claro.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def denoise(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apertura morfológica para eliminar ruido aislado (sal)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def crop_to_bounding_box(binary: np.ndarray, margin: int = 5) -> np.ndarray:
    """Recorta la imagen al bounding box del contenido (firma)."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = binary.shape
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)
    return binary[y0:y1, x0:x1]


def resize_with_aspect_ratio(
    img: np.ndarray, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """Redimensiona conservando aspect ratio y rellena con ceros."""
    target_w, target_h = target_size
    h, w = img.shape

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess(
    image: str | Path | np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Pipeline completo de preprocesamiento."""
    if isinstance(image, (str, Path)):
        gray = load_image(image)
    else:
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    binary = binarize_otsu(gray)
    cleaned = denoise(binary)
    cropped = crop_to_bounding_box(cleaned)
    normalized = resize_with_aspect_ratio(cropped, target_size)
    return normalized


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m src.preprocessing <ruta_imagen>")
        sys.exit(1)

    out = preprocess(sys.argv[1])
    print(f"Imagen procesada: shape={out.shape}, dtype={out.dtype}")
    print(f"Píxeles activos: {int((out > 0).sum())}")
    cv2.imwrite("preprocessed.png", out)
    print("Guardado en preprocessed.png")
