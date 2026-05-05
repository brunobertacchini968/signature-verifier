"""Alineación geométrica de dos firmas mediante transformación afín.

Estrategia:
    1. ORB detecta keypoints en ambas imágenes.
    2. BFMatcher empareja descriptores con ratio test de Lowe.
    3. RANSAC estima la matriz afín 2x3 que mapea source -> target.
    4. cv2.warpAffine aplica la transformación a la imagen source.

Devuelve la imagen alineada, la matriz A, el vector de traslación,
los keypoints inliers y el ratio de inliers (feature reutilizado en
features.py como señal de similitud).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_ORB_FEATURES = 1000
LOWE_RATIO = 0.75
MIN_INLIERS_FOR_VALID_ALIGNMENT = 8


@dataclass
class AlignmentResult:
    aligned: np.ndarray            # imagen source alineada al target
    matrix: np.ndarray | None      # matriz afín 2x3 (None si falla RANSAC)
    inlier_ratio: float            # inliers / total matches buenos
    num_good_matches: int
    num_inliers: int
    success: bool


def detect_and_describe(
    image: np.ndarray, n_features: int = DEFAULT_ORB_FEATURES
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detecta keypoints ORB y computa descriptores binarios."""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return list(keypoints), descriptors


def match_descriptors(
    desc_source: np.ndarray,
    desc_target: np.ndarray,
    ratio: float = LOWE_RATIO,
) -> list[cv2.DMatch]:
    """Empareja descriptores con BFMatcher + ratio test de Lowe.

    Para descriptores binarios ORB se usa NORM_HAMMING. KnnMatch con k=2
    permite filtrar matches ambiguos comparando primer y segundo vecino.
    """
    if desc_source is None or desc_target is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc_source, desc_target, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_affine(
    kp_source: list[cv2.KeyPoint],
    kp_target: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estima matriz afín 2x3 con RANSAC. Devuelve (M, mask_inliers)."""
    if len(matches) < MIN_INLIERS_FOR_VALID_ALIGNMENT:
        return None, None

    src_pts = np.float32(
        [kp_source[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_target[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # estimateAffinePartial2D restringe a translation+rotation+uniform scale.
    # estimateAffine2D permite shear y anisotropic scaling -> mejor para Das.
    matrix, inlier_mask = cv2.estimateAffine2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    return matrix, inlier_mask


def align(
    source: np.ndarray, target: np.ndarray
) -> AlignmentResult:
    """Alinea source para que matchee target. Pipeline completo."""
    kp_s, desc_s = detect_and_describe(source)
    kp_t, desc_t = detect_and_describe(target)
    good_matches = match_descriptors(desc_s, desc_t)

    matrix, inlier_mask = estimate_affine(kp_s, kp_t, good_matches)

    if matrix is None:
        return AlignmentResult(
            aligned=source.copy(),
            matrix=None,
            inlier_ratio=0.0,
            num_good_matches=len(good_matches),
            num_inliers=0,
            success=False,
        )

    h, w = target.shape
    aligned = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_LINEAR)

    num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = (
        num_inliers / len(good_matches) if good_matches else 0.0
    )

    return AlignmentResult(
        aligned=aligned,
        matrix=matrix,
        inlier_ratio=inlier_ratio,
        num_good_matches=len(good_matches),
        num_inliers=num_inliers,
        success=num_inliers >= MIN_INLIERS_FOR_VALID_ALIGNMENT,
    )


if __name__ == "__main__":
    import sys

    from .preprocessing import preprocess

    if len(sys.argv) < 3:
        print("Uso: python -m src.alignment <imagen_a> <imagen_b>")
        sys.exit(1)

    a = preprocess(sys.argv[1])
    b = preprocess(sys.argv[2])
    result = align(a, b)
    print(f"Success: {result.success}")
    print(f"Good matches: {result.num_good_matches}")
    print(f"Inliers: {result.num_inliers}")
    print(f"Inlier ratio: {result.inlier_ratio:.3f}")
    if result.matrix is not None:
        print(f"Matriz afín:\n{result.matrix}")
