"""Experimento 2: ablation study.

Lee los scores y features ya computados en exp1 (results/exp1/scores.csv) y
recalcula el EER quitando un bloque de features cada vez. Esto justifica
empíricamente la inclusión de cada feature en el vector final.

Bloques:
    structural   = ssim, xor_iou, hu_distance
    geometric    = anisotropic_scaling, residual_error, ransac_inlier_ratio, simple_geometric
    texture      = glcm_distance
    directional  = chain_code_distance
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.evaluator import evaluate
from src.features import FEATURE_NAMES
from src.verifier import normalize_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = PROJECT_ROOT / "results" / "exp1" / "scores.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp2"


FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "structural":  ("ssim", "xor_iou", "hu_distance"),
    "geometric":   (
        "anisotropic_scaling",
        "residual_error",
        "ransac_inlier_ratio",
        "simple_geometric",
    ),
    "texture":     ("glcm_distance",),
    "directional": ("chain_code_distance",),
}


def _recompute_score(row: dict, mask: np.ndarray) -> float:
    """Recalcula el score normalizado promediando solo los features activos."""
    raw = np.array([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
    normalized = normalize_features(raw)
    if mask.sum() == 0:
        return 0.0
    weights = mask.astype(np.float64) / mask.sum()
    return float(np.dot(weights, normalized))


def run() -> None:
    if not SCORES_CSV.exists():
        print(f"ERROR: no existe {SCORES_CSV}. Corré primero exp1_baseline.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with SCORES_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("ERROR: scores.csv está vacío.")
        return

    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    print("=== EXP 2: ABLATION STUDY ===\n")
    print(f"{'Configuración':<30} {'EER':>10} {'AUC':>10}")
    print("-" * 52)

    summary_lines = [f"{'Configuración':<30} {'EER':>10} {'AUC':>10}", "-" * 52]

    # Baseline: todos los features.
    full_mask = np.ones(len(FEATURE_NAMES), dtype=bool)
    sg = []
    sf = []
    for r in rows:
        score = _recompute_score(r, full_mask)
        if r["label"] == "genuine":
            sg.append(score)
        else:
            sf.append(score)
    metrics = evaluate(np.array(sg), np.array(sf))
    line = f"{'all (baseline)':<30} {metrics.eer:>10.4f} {metrics.auc:>10.4f}"
    print(line)
    summary_lines.append(line)

    # Quitar cada bloque por turno.
    for block_name, block_features in FEATURE_BLOCKS.items():
        mask = full_mask.copy()
        for fname in block_features:
            mask[name_to_idx[fname]] = False
        sg, sf = [], []
        for r in rows:
            score = _recompute_score(r, mask)
            if r["label"] == "genuine":
                sg.append(score)
            else:
                sf.append(score)
        metrics = evaluate(np.array(sg), np.array(sf))
        line = (
            f"{'sin ' + block_name:<30} "
            f"{metrics.eer:>10.4f} {metrics.auc:>10.4f}"
        )
        print(line)
        summary_lines.append(line)

    # Cada bloque solo (para ver poder discriminativo individual).
    print()
    summary_lines.append("")
    print("Cada bloque por separado:")
    summary_lines.append("Cada bloque por separado:")
    for block_name, block_features in FEATURE_BLOCKS.items():
        mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
        for fname in block_features:
            mask[name_to_idx[fname]] = True
        sg, sf = [], []
        for r in rows:
            score = _recompute_score(r, mask)
            if r["label"] == "genuine":
                sg.append(score)
            else:
                sf.append(score)
        metrics = evaluate(np.array(sg), np.array(sf))
        line = (
            f"{'solo ' + block_name:<30} "
            f"{metrics.eer:>10.4f} {metrics.auc:>10.4f}"
        )
        print(line)
        summary_lines.append(line)

    (RESULTS_DIR / "summary.txt").write_text("\n".join(summary_lines))
    print(f"\nResultados guardados en {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
