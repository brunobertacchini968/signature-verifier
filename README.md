# Signature Verifier

Sistema de verificación de firmas offline (offline signature verification) usando procesamiento clásico de imágenes y transformaciones geométricas. Trabajo final de materia universitaria.

## Objetivo

Dado un par de imágenes de firmas, decidir si pertenecen a la misma persona y devolver un score de matching en porcentaje. Sistema biométrico evaluado con métricas estándar (FAR, FRR, EER, ROC) sobre el dataset CEDAR.

## Enfoque

Procesamiento clásico de imágenes — sin redes neuronales. El núcleo es la **alineación geométrica** de las dos firmas mediante transformaciones afines estimadas por ORB + RANSAC, seguida de extracción de un vector de 9 features y combinación ponderada en un score final.

## Pipeline

1. **Preprocesamiento** — grayscale, binarización Otsu, denoise morfológico, recorte al bounding box, normalización de tamaño.
2. **Alineación geométrica** — ORB keypoints + BFMatcher + RANSAC → matriz afín 2×3 → `warpAffine`.
3. **Extracción de 9 features** en 4 categorías:
   - Estructural: SSIM, XOR ratio, distancia entre Hu Moments
   - Geométrica: anisotropic scaling (Das), residual error (Dre), RANSAC inlier ratio, geométricos simples
   - Textura: GLCM (contraste, correlación, energía, homogeneidad, entropía)
   - Direccional: histograma de chain codes
4. **Score final** — combinación ponderada (pesos optimizados con regresión logística minimizando EER).

## Estructura

```
signature-verifier/
├── data/                  # CEDAR dataset (no versionado)
├── src/
│   ├── preprocessing.py   # binarización, crop, normalización
│   ├── alignment.py       # ORB + RANSAC + warpAffine
│   ├── features.py        # 9 features
│   ├── verifier.py        # combina features → score
│   ├── evaluator.py       # FAR/FRR/EER/ROC
│   └── perturbations.py   # rotación, ruido, escala, oclusión
├── experiments/
│   ├── exp1_baseline.py
│   ├── exp2_ablation.py
│   └── exp3_robustness.py
├── notebooks/
├── report/
└── tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

CEDAR Signature Database (University at Buffalo). 55 firmantes, 24 firmas genuinas + 24 falsificaciones hábiles por persona. Descargar y descomprimir en `data/`:

```
data/
├── genuine/
│   ├── original_1_1.png ... original_1_24.png
│   └── ...
└── skilled_forgery/
    ├── forgeries_1_1.png ... forgeries_1_24.png
    └── ...
```

Los pares de falsificación aleatoria se generan en runtime cruzando firmantes distintos.

## Ejecución

Experimentos:

```bash
python -m experiments.exp1_baseline
python -m experiments.exp2_ablation
python -m experiments.exp3_robustness
```

Tests unitarios:

```bash
pytest tests/
```

## Equipo

3 integrantes. División de trabajo:
- Pipeline de imagen: `preprocessing.py`, `alignment.py`
- Features y matching: `features.py`, `verifier.py`
- Evaluación y experimentos: dataset, `evaluator.py`, `perturbations.py`

## Bibliografía

- Zhu, Zheng, Doermann, Jaeger (2009). *Signature Detection and Matching for Document Image Retrieval*. IEEE TPAMI 31(11).
- Hameed et al. (2021). *Machine Learning-Based Offline Signature Verification Systems: A Systematic Review*. Signal Processing: Image Communication.
- Batool et al. (2020). *Offline Signature Verification System: A Novel Technique of Fusion of GLCM and Geometric Features Using SVM*. Multimedia Tools and Applications.
- Rivard et al. (2021). *Off-Line Signature Verification Using Elementary Combinations of Directional Codes from Boundary Pixels*. Neural Computing and Applications.
