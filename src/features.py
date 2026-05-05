"""Extracción de features para comparar dos firmas alineadas.

Vector de 9 features en 4 categorías:

ESTRUCTURAL
  1. SSIM (Structural Similarity Index)
  2. XOR ratio (intersección sobre unión de píxeles activos)
  3. Hu Moments distance (log-distancia entre momentos invariantes)

GEOMÉTRICA
  4. Das  - anisotropic scaling: log(max(Sx,Sy)/min(Sx,Sy)) sobre SVD de A.
           Capta cuánta deformación direccional hizo falta para alinear.
           Idea de Zhu et al. 2009 TPAMI.
  5. Dre  - registration residual error: distancia euclídea promedio entre
           keypoints emparejados después de aplicar la transformación.
  6. RANSAC inlier ratio (ya calculado en alignment).
  9. Geométricos simples: aspect ratio diff, occupancy diff, baseline angle
     diff, stroke count diff (agregados al vector).

TEXTURA
  7. GLCM features (5 propiedades de Haralick): contraste, correlación,
     energía, homogeneidad, entropía. Distancia entre vectores.

DIRECCIONAL
  8. Chain code direction histogram distance (8 direcciones de Freeman).

Cada feature_xxx() devuelve un escalar. compute_features() devuelve un
dict con todos los nombres y valores, más una versión vector (np.array).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim


# ============================================================
# ESTRUCTURAL
# ============================================================

def feature_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """SSIM entre dos imágenes binarias. Rango [-1, 1], típicamente [0, 1]."""
    return float(ssim(img_a, img_b, data_range=255))


def feature_xor_ratio(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Jaccard / IoU entre los píxeles activos. Rango [0, 1]."""
    a_bin = img_a > 0
    b_bin = img_b > 0
    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _hu_log(img: np.ndarray) -> np.ndarray:
    """Hu Moments en escala log para que sean comparables."""
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # log-transform: -sign(h) * log10(|h|+eps), siguiendo convención clásica.
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def feature_hu_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Distancia euclídea entre los Hu Moments (en log-space)."""
    hu_a = _hu_log(img_a)
    hu_b = _hu_log(img_b)
    return float(np.linalg.norm(hu_a - hu_b))


# ============================================================
# GEOMÉTRICA
# ============================================================

def feature_anisotropic_scaling(matrix: np.ndarray | None) -> float:
    """Das = log(max(Sx,Sy)/min(Sx,Sy)) sobre SVD de la submatriz 2x2.

    0 cuando el escalado es isotrópico (firmas afines pura rotación+escala
    uniforme). Crece cuando hizo falta estirar mucho un eje vs el otro.
    """
    if matrix is None:
        return float("inf")
    A = matrix[:, :2]
    sigmas = np.linalg.svd(A, compute_uv=False)
    if np.min(sigmas) < 1e-10:
        return float("inf")
    return float(np.log(np.max(sigmas) / np.min(sigmas)))


def feature_residual_error(
    matrix: np.ndarray | None,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> float:
    """Promedio de distancia euclídea entre puntos transformados y target."""
    if matrix is None or len(src_pts) == 0:
        return float("inf")
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    ones = np.ones((src.shape[0], 1))
    src_h = np.hstack([src, ones])
    transformed = (matrix @ src_h.T).T
    residuals = np.linalg.norm(transformed - dst, axis=1)
    return float(np.mean(residuals))


# ============================================================
# TEXTURA - GLCM
# ============================================================

GLCM_PROPS = ("contrast", "correlation", "energy", "homogeneity")


def _glcm_vector(img: np.ndarray) -> np.ndarray:
    """Computa 5 propiedades GLCM (4 de Haralick + entropía) sobre la imagen."""
    # GLCM requiere uint8; usar 4 ángulos y distancia 1.
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = [graycoprops(glcm, p).mean() for p in GLCM_PROPS]
    # Entropía manual (skimage no la trae directamente).
    p = glcm[..., 0, :]
    p_nonzero = p[p > 0]
    entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
    props.append(entropy)
    return np.array(props, dtype=np.float64)


def feature_glcm_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Distancia euclídea normalizada entre vectores GLCM."""
    va = _glcm_vector(img_a)
    vb = _glcm_vector(img_b)
    # Normalización por escala de cada propiedad para que no domine alguna.
    scale = np.maximum(np.abs(va) + np.abs(vb), 1e-6)
    return float(np.linalg.norm((va - vb) / scale))


# ============================================================
# DIRECCIONAL - Chain codes
# ============================================================

# Vectores de los 8 vecinos (chain code de Freeman).
FREEMAN_DIRECTIONS = np.array(
    [
        (1, 0),    # 0: E
        (1, -1),   # 1: NE
        (0, -1),   # 2: N
        (-1, -1),  # 3: NW
        (-1, 0),   # 4: W
        (-1, 1),   # 5: SW
        (0, 1),    # 6: S
        (1, 1),    # 7: SE
    ]
)


def _chain_code_histogram(img: np.ndarray) -> np.ndarray:
    """Histograma de 8 direcciones sobre los contornos de la firma."""
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    histogram = np.zeros(8, dtype=np.float64)
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue
        diffs = np.diff(pts, axis=0)
        # Cada diff es un step de un pixel; mapear a la dirección más cercana.
        for dx, dy in diffs:
            # dy invertido porque en imagen y crece hacia abajo.
            best = np.argmax(
                FREEMAN_DIRECTIONS @ np.array([dx, -dy], dtype=np.float64)
            )
            histogram[best] += 1
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def feature_chain_code_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Distancia chi-cuadrado entre histogramas de direcciones."""
    ha = _chain_code_histogram(img_a)
    hb = _chain_code_histogram(img_b)
    denom = ha + hb + 1e-10
    return float(0.5 * np.sum(((ha - hb) ** 2) / denom))


# ============================================================
# GEOMÉTRICOS SIMPLES
# ============================================================

def _shape_descriptors(img: np.ndarray) -> dict:
    """Aspect ratio, occupancy, baseline angle, stroke count."""
    coords = cv2.findNonZero(img)
    if coords is None:
        return {
            "aspect_ratio": 0.0,
            "occupancy": 0.0,
            "baseline_angle": 0.0,
            "stroke_count": 0,
        }
    x, y, w, h = cv2.boundingRect(coords)
    area = (img > 0).sum()
    bbox_area = max(1, w * h)
    aspect_ratio = w / max(1, h)
    occupancy = area / bbox_area

    # Baseline angle por PCA sobre los píxeles activos.
    pts = coords.reshape(-1, 2).astype(np.float64)
    if len(pts) >= 2:
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        baseline_angle = float(np.arctan2(principal[1], principal[0]))
    else:
        baseline_angle = 0.0

    n_components, _ = cv2.connectedComponents(img)
    stroke_count = max(0, n_components - 1)  # el componente 0 es el fondo

    return {
        "aspect_ratio": float(aspect_ratio),
        "occupancy": float(occupancy),
        "baseline_angle": baseline_angle,
        "stroke_count": stroke_count,
    }


def feature_simple_geometric_distance(
    img_a: np.ndarray, img_b: np.ndarray
) -> float:
    """Distancia entre los descriptores geométricos simples normalizados."""
    da = _shape_descriptors(img_a)
    db = _shape_descriptors(img_b)
    diffs = np.array(
        [
            abs(da["aspect_ratio"] - db["aspect_ratio"]),
            abs(da["occupancy"] - db["occupancy"]),
            abs(da["baseline_angle"] - db["baseline_angle"]),
            abs(da["stroke_count"] - db["stroke_count"])
            / max(1, max(da["stroke_count"], db["stroke_count"])),
        ]
    )
    return float(np.linalg.norm(diffs))


# ============================================================
# AGREGADOR
# ============================================================

FEATURE_NAMES = (
    "ssim",                   # estructural - similitud (mayor = más parecidas)
    "xor_iou",                # estructural - similitud
    "hu_distance",            # estructural - distancia (menor = más parecidas)
    "anisotropic_scaling",    # geométrica  - distancia
    "residual_error",         # geométrica  - distancia
    "ransac_inlier_ratio",    # geométrica  - similitud
    "glcm_distance",          # textura     - distancia
    "chain_code_distance",    # direccional - distancia
    "simple_geometric",       # geométrica  - distancia
)


@dataclass
class FeatureVector:
    values: np.ndarray  # shape (9,)
    names: tuple[str, ...] = FEATURE_NAMES

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist()))


def compute_features(
    img_a_aligned: np.ndarray,
    img_b: np.ndarray,
    alignment_matrix: np.ndarray | None,
    inlier_ratio: float,
    src_pts: np.ndarray | None = None,
    dst_pts: np.ndarray | None = None,
) -> FeatureVector:
    """Extrae los 9 features dado un par alineado y metadatos del alineamiento."""
    f_ssim = feature_ssim(img_a_aligned, img_b)
    f_xor = feature_xor_ratio(img_a_aligned, img_b)
    f_hu = feature_hu_distance(img_a_aligned, img_b)
    f_das = feature_anisotropic_scaling(alignment_matrix)
    f_dre = (
        feature_residual_error(alignment_matrix, src_pts, dst_pts)
        if src_pts is not None and dst_pts is not None
        else 0.0
    )
    f_inlier = float(inlier_ratio)
    f_glcm = feature_glcm_distance(img_a_aligned, img_b)
    f_chain = feature_chain_code_distance(img_a_aligned, img_b)
    f_geom = feature_simple_geometric_distance(img_a_aligned, img_b)

    return FeatureVector(
        values=np.array(
            [f_ssim, f_xor, f_hu, f_das, f_dre, f_inlier, f_glcm, f_chain, f_geom],
            dtype=np.float64,
        )
    )
