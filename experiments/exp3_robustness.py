"""Experimento 3: análisis de robustez.

Aplica perturbaciones controladas a una de las imágenes de cada par y
recalcula el EER. Tabla resultante muestra cómo degrada el sistema bajo
cada tipo y nivel de perturbación.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.dataset import build_pairs, load_cedar
from src.evaluator import evaluate
from src.perturbations import PERTURBATION_GRID, apply_perturbation
from src.verifier import SignatureVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp3"


def _verify_with_perturbation(
    verifier: SignatureVerifier,
    path_a: Path,
    path_b: Path,
    perturb_fn,
    perturb_kwargs: dict,
) -> float:
    """Carga A, perturba A, verifica contra B (sin perturbar). Devuelve score."""
    img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return float("nan")
    img_a_perturbed = apply_perturbation(img_a, perturb_fn, **perturb_kwargs)
    result = verifier.verify(img_a_perturbed, img_b)
    return result.score_pct / 100.0


def run(n_per_category: int = 200) -> None:
    """n_per_category más bajo que exp1 porque corremos N veces (1 por perturbación)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset CEDAR...")
    genuines, forgeries = load_cedar(DATA_ROOT)
    if not genuines or not forgeries:
        print(f"ERROR: no se encontraron firmas en {DATA_ROOT}.")
        return

    pairs = build_pairs(genuines, forgeries, n_per_category=n_per_category)
    print(f"Pares generados: {len(pairs)}")

    verifier = SignatureVerifier()

    rows = []
    print("\n=== EXP 3: ROBUSTEZ ===\n")
    print(f"{'Perturbación':<20} {'EER':>10} {'AUC':>10} {'ΔEER vs baseline':>20}")
    print("-" * 65)

    # Baseline: sin perturbación.
    sg, sf = [], []
    for pair in tqdm(pairs, desc="Baseline"):
        result = verifier.verify(pair.a, pair.b)
        score = result.score_pct / 100.0
        if pair.label == "genuine":
            sg.append(score)
        else:
            sf.append(score)
    base_metrics = evaluate(np.array(sg), np.array(sf))
    base_line = f"{'baseline':<20} {base_metrics.eer:>10.4f} {base_metrics.auc:>10.4f} {'-':>20}"
    print(base_line)
    rows.append({
        "perturbation": "baseline",
        "eer": base_metrics.eer,
        "auc": base_metrics.auc,
        "delta_eer": 0.0,
    })

    # Cada perturbación.
    for category, variants in PERTURBATION_GRID.items():
        for name, fn, kwargs in variants:
            sg, sf = [], []
            for pair in tqdm(pairs, desc=name, leave=False):
                score = _verify_with_perturbation(verifier, pair.a, pair.b, fn, kwargs)
                if np.isnan(score):
                    continue
                if pair.label == "genuine":
                    sg.append(score)
                else:
                    sf.append(score)
            if not sg or not sf:
                continue
            metrics = evaluate(np.array(sg), np.array(sf))
            delta = metrics.eer - base_metrics.eer
            line = f"{name:<20} {metrics.eer:>10.4f} {metrics.auc:>10.4f} {delta:>+20.4f}"
            print(line)
            rows.append({
                "perturbation": name,
                "category": category,
                "eer": metrics.eer,
                "auc": metrics.auc,
                "delta_eer": delta,
            })

    csv_path = RESULTS_DIR / "robustness.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["perturbation", "category", "eer", "auc", "delta_eer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r.setdefault("category", "")
            writer.writerow(r)
    print(f"\nResultados guardados en {csv_path}")


if __name__ == "__main__":
    run()
