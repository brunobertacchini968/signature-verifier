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
import matplotlib.pyplot as plt

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
        show_plot: bool = False  # <-- Agregamos esta opción
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
        score_pct = score * 100.0

        decision = "GENUINE" if score >= self.config.threshold else "FORGERY"

        # --- Llamamos a la visualización gráfica aquí ---
        if show_plot:
            show_matches(a, b, alignment, score_pct)

        return VerificationResult(
            score_pct=score_pct,
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


def show_matches(img_source, img_target, align_result, score_porcentaje_total):
    """
    Muestra visualización detallada: cuadrados verdes en inliers bien distribuidos
    y porcentajes locales. Evita la saturación de líneas que oculta la firma.
    """
    # 1. Verificar datos necesarios
    if not align_result.kp_source or not align_result.kp_target or not align_result.matches:
        print("Faltan datos geométricos (keypoints/matches) para graficar.")
        return

    # 2. Preparar el lienzo (montaje horizontal manual)
    if len(img_source.shape) == 2:
        canvas_a = cv2.cvtColor(img_source, cv2.COLOR_GRAY2BGR)
    else:
        canvas_a = img_source.copy()

    if len(img_target.shape) == 2:
        canvas_b = cv2.cvtColor(img_target, cv2.COLOR_GRAY2BGR)
    else:
        canvas_b = img_target.copy()

    h_a, w_a = canvas_a.shape[:2]
    h_b, w_b = canvas_b.shape[:2]
    out_img = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype=np.uint8)
    out_img[:h_a, :w_a, :] = canvas_a
    out_img[:h_b, w_a:w_a + w_b, :] = canvas_b

    # 3. Filtrar Inliers (coincidencias validadas por RANSAC)
    good_matches = []
    if align_result.inlier_mask is not None:
        for i, match in enumerate(align_result.matches):
            if align_result.inlier_mask[i][0]:
                good_matches.append(match)
    else:
        good_matches = align_result.matches

    # Ordenar por calidad (menor distancia Hamming es mejor coincidencia)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # --- Lógica de Supresión Espacial (Evitar que se encimen) ---
    MAX_VISUAL_MATCHES = 8  # Mostrar máximo 8 cuadros
    MIN_SPATIAL_DIST = 45  # Separación obligatoria de 45 píxeles entre cuadros

    matches_to_draw = []
    puntos_ya_dibujados = []
    kp_a = align_result.kp_source
    kp_b = align_result.kp_target

    for match in good_matches:
        pt_actual = np.array(kp_a[match.queryIdx].pt)

        # Revisar si choca con los que ya pusimos
        muy_cerca = False
        for pt_dibujado in puntos_ya_dibujados:
            if np.linalg.norm(pt_actual - pt_dibujado) < MIN_SPATIAL_DIST:
                muy_cerca = True
                break

        # Si tiene espacio libre, lo agregamos a la lista de dibujo
        if not muy_cerca:
            matches_to_draw.append(match)
            puntos_ya_dibujados.append(pt_actual)

        # Si llegamos al máximo de 8 cuadros, dejamos de buscar
        if len(matches_to_draw) >= MAX_VISUAL_MATCHES:
            break

    # 4. Bucle de dibujo de cuadrados, líneas y textos
    box_size = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text_color = (0, 255, 0)
    text_offset_y = 5

    for match in matches_to_draw:
        pt_a = tuple(np.round(kp_a[match.queryIdx].pt).astype(int))
        pt_b = tuple(np.round(kp_b[match.trainIdx].pt).astype(int))
        pt_b_shifted = (pt_b[0] + w_a, pt_b[1])

        # A. Calcular porcentaje local (Distancia ORB convertida a 0-100%)
        MAX_DIST_FOR_0_PCT = 75
        local_score_norm = max(0, (1.0 - (match.distance / MAX_DIST_FOR_0_PCT)))
        local_score_pct = local_score_norm * 100.0

        # B. Dibujar Cuadrados
        tl_a = (pt_a[0] - box_size // 2, pt_a[1] - box_size // 2)
        br_a = (pt_a[0] + box_size // 2, pt_a[1] + box_size // 2)
        cv2.rectangle(out_img, tl_a, br_a, (0, 255, 0), thickness=2)

        tl_b = (pt_b_shifted[0] - box_size // 2, pt_b_shifted[1] - box_size // 2)
        br_b = (pt_b_shifted[0] + box_size // 2, pt_b_shifted[1] + box_size // 2)
        cv2.rectangle(out_img, tl_b, br_b, (0, 255, 0), thickness=2)

        # C. Dibujar Línea de unión finita
        cv2.line(out_img, pt_a, pt_b_shifted, (0, 150, 0), thickness=1)

        # D. Dibujar Textos de porcentaje
        text = f"{local_score_pct:.0f}%"
        cv2.putText(out_img, text, (tl_a[0], tl_a[1] - text_offset_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(out_img, text, (tl_b[0], tl_b[1] - text_offset_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 5. Visualizar con Matplotlib
    out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 7))
    plt.imshow(out_img_rgb)

    color_titulo = 'green' if score_porcentaje_total >= 75.0 else 'red'
    plt.title(f"Coincidencia Total: {score_porcentaje_total:.2f}%", fontsize=18, fontweight='bold', color=color_titulo)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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

    # Creamos la instancia y le pasamos show_plot=True
    verifier = SignatureVerifier()
    result = verifier.verify(sys.argv[1], sys.argv[2], show_plot=True)

    print(f"Score: {result.score_pct:.2f}%  ->  {result.decision}")
    print("Features (raw):")
    for k, v in result.raw_features.as_dict().items():
        print(f"  {k:25s} {v:.4f}")
