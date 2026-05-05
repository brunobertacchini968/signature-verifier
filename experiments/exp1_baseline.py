"""Experimento 1: baseline del sistema sobre CEDAR.

Calcula score para todos los pares (genuine / skilled_forgery / random_forgery),
genera histogramas, ROC, FAR/FRR, EER y umbral óptimo.

Salida:
    results/exp1/scores.csv
    results/exp1/roc.png
    results/exp1/far_frr.png
    results/exp1/histogram.png
    results/exp1/summary.txt
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.dataset import build_pairs, load_cedar
from src.evaluator import (
    evaluate,
    plot_far_frr,
    plot_roc,
    plot_score_histograms,
    summary,
)
from src.verifier import SignatureVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp1"


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset CEDAR...")
    genuines, forgeries = load_cedar(DATA_ROOT)
    print(f"  Genuinas:        {len(genuines)}")
    print(f"  Falsificaciones: {len(forgeries)}")

    if not genuines or not forgeries:
        print(f"ERROR: no se encontraron firmas en {DATA_ROOT}.")
        print("Descargá CEDAR y ubicá las imágenes en data/genuine/ y data/skilled_forgery/.")
        return

    pairs = build_pairs(genuines, forgeries, n_per_category=500)
    print(f"Pares generados: {len(pairs)}")

    verifier = SignatureVerifier()
    scores_per_label: dict[str, list[float]] = {
        "genuine": [],
        "skilled_forgery": [],
        "random_forgery": [],
    }

    rows: list[dict] = []
    for pair in tqdm(pairs, desc="Verificando pares"):
        try:
            result = verifier.verify(pair.a, pair.b)
            score = result.score_pct / 100.0
            scores_per_label[pair.label].append(score)
            rows.append({
                "a": str(pair.a.relative_to(PROJECT_ROOT)),
                "b": str(pair.b.relative_to(PROJECT_ROOT)),
                "label": pair.label,
                "score": score,
                "decision": result.decision,
                "alignment_success": result.alignment_success,
                **result.raw_features.as_dict(),
            })
        except Exception as e:
            print(f"  Falló par ({pair.a.name}, {pair.b.name}): {e}")

    # Persistir scores.
    csv_path = RESULTS_DIR / "scores.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Scores guardados en {csv_path}")

    # Evaluación contra ambos tipos de falsificación combinados y por separado.
    sg = np.array(scores_per_label["genuine"])
    sf_skilled = np.array(scores_per_label["skilled_forgery"])
    sf_random = np.array(scores_per_label["random_forgery"])
    sf_all = np.concatenate([sf_skilled, sf_random]) if len(sf_skilled) + len(sf_random) else np.array([])

    if len(sg) == 0 or len(sf_all) == 0:
        print("ERROR: no se obtuvieron scores suficientes para evaluar.")
        return

    metrics_all = evaluate(sg, sf_all)
    metrics_skilled = evaluate(sg, sf_skilled) if len(sf_skilled) else None
    metrics_random = evaluate(sg, sf_random) if len(sf_random) else None

    summary_lines = ["=== EXP 1: BASELINE ===", "", "Genuinas vs todas las falsificaciones:"]
    summary_lines.append(summary(metrics_all))
    if metrics_skilled:
        summary_lines += ["", "Genuinas vs falsificación HÁBIL:", summary(metrics_skilled)]
    if metrics_random:
        summary_lines += ["", "Genuinas vs falsificación ALEATORIA:", summary(metrics_random)]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)
    (RESULTS_DIR / "summary.txt").write_text(summary_text)

    # Gráficos.
    plot_roc(metrics_all, save_to=RESULTS_DIR / "roc.png")
    plot_far_frr(metrics_all, save_to=RESULTS_DIR / "far_frr.png")
    plot_score_histograms(sg, sf_all, save_to=RESULTS_DIR / "histogram.png")
    print(f"Gráficos guardados en {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
