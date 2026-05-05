"""Verificador de firmas: orquesta preprocesamiento + alineación + features
+ combinador de score.

Devuelve un porcentaje de matching en [0, 100]. Internamente calcula el
vector de 9 features y los combina con pesos optimizados (regresión
logística) o con pesos uniformes si no se proveen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .alignment import align, detect_and_describe, match_descriptors
from .features import FEATURE_NAMES, FeatureVector, compute_features
from .preprocessing import preprocess


# Sign convention: queremos que cada feature, después de normalizar,
# crezca con la similitud. Para distancias hay que invertirlas.
# True  -> ya es similitud (mayor = más parecidas)
# False -> es distancia (mayor = más distintas)  -> se invierte con exp(-x)
FEATURE_IS_SIMILARITY = {
    "ssim": True,
    "xor_iou": True,
    "hu_distance": False,
    "anisotropic_scaling": False,
    "residual_error": False,
    "ransac_inlier_ratio": True,
    "glcm_distance": False,
    "chain_code_distance": False,
    "simple_geometric": False,
}


def normalize_features(values: np.ndarray) -> np.ndarray:
    """Pasa cada feature a [0,1] como similitud (1 = idéntico)."""
    out = np.zeros_like(values)
    for i, name in enumerate(FEATURE_NAMES):
        v = values[i]
        if not np.isfinite(v):
            out[i] = 0.0
            continue
        if FEATURE_IS_SIMILARITY[name]:
            out[i] = float(np.clip(v, 0.0, 1.0))
        else:
            # Distancia -> similitud por kernel exponencial.
            out[i] = float(np.exp(-v))
    return out


@dataclass
class VerifierConfig:
    weights: np.ndarray = field(
        default_factory=lambda: np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)
    )
    threshold: float = 0.5  # umbral en el score normalizado [0,1]


@dataclass
class VerificationResult:
    score_pct: float           # porcentaje 0-100
    decision: str              # "GENUINE" | "FORGERY"
    raw_features: FeatureVector
    normalized_features: np.ndarray
    alignment_success: bool


class SignatureVerifier:
    """API de alto nivel para comparar dos firmas."""

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()

    def verify(
        self,
        image_a: str | Path | np.ndarray,
        image_b: str | Path | np.ndarray,
    ) -> VerificationResult:
        a = preprocess(image_a)
        b = preprocess(image_b)

        alignment = align(a, b)

        # Recolectar matches en arrays para el feature de residual.
        src_pts, dst_pts = self._extract_inlier_points(a, b, alignment)

        features = compute_features(
            img_a_aligned=alignment.aligned,
            img_b=b,
            alignment_matrix=alignment.matrix,
            inlier_ratio=alignment.inlier_ratio,
            src_pts=src_pts,
            dst_pts=dst_pts,
        )

        normalized = normalize_features(features.values)
        score = float(np.dot(self.config.weights, normalized))
        score = float(np.clip(score, 0.0, 1.0))

        decision = "GENUINE" if score >= self.config.threshold else "FORGERY"

        return VerificationResult(
            score_pct=score * 100.0,
            decision=decision,
            raw_features=features,
            normalized_features=normalized,
            alignment_success=alignment.success,
        )

    @staticmethod
    def _extract_inlier_points(
        a: np.ndarray, b: np.ndarray, alignment
    ) -> tuple[np.ndarray, np.ndarray]:
        if not alignment.success:
            return np.empty((0, 2)), np.empty((0, 2))
        kp_a, desc_a = detect_and_describe(a)
        kp_b, desc_b = detect_and_describe(b)
        matches = match_descriptors(desc_a, desc_b)
        if not matches:
            return np.empty((0, 2)), np.empty((0, 2))
        src = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float64)
        dst = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float64)
        return src, dst


def verify(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
    config: VerifierConfig | None = None,
) -> VerificationResult:
    """Atajo funcional."""
    return SignatureVerifier(config).verify(image_a, image_b)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python -m src.verifier <firma_a> <firma_b>")
        sys.exit(1)

    result = verify(sys.argv[1], sys.argv[2])
    print(f"Score: {result.score_pct:.2f}%  ->  {result.decision}")
    print("Features (raw):")
    for k, v in result.raw_features.as_dict().items():
        print(f"  {k:25s} {v:.4f}")
