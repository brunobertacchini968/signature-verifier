"""Evaluación del verificador como sistema biométrico.

Métricas:
    FAR (False Acceptance Rate)  - falsificaciones aceptadas / total falsificaciones.
    FRR (False Rejection Rate)   - genuinas rechazadas / total genuinas.
    EER (Equal Error Rate)       - punto donde FAR = FRR. Métrica única estándar.
    ROC + AUC                    - desempeño en todos los umbrales.

Uso:
    scores_genuine, scores_forgery = ...  # arrays de scores en [0,1]
    metrics = evaluate(scores_genuine, scores_forgery)
    plot_roc(metrics, save_to="results/roc.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalMetrics:
    thresholds: np.ndarray
    far: np.ndarray              # mismo tamaño que thresholds
    frr: np.ndarray
    tpr: np.ndarray              # = 1 - frr  (para ROC)
    fpr: np.ndarray              # = far      (para ROC)
    eer: float
    eer_threshold: float
    auc: float
    n_genuine: int
    n_forgery: int


def _compute_far_frr(
    scores_genuine: np.ndarray,
    scores_forgery: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Para cada umbral, calcula FAR y FRR."""
    far = np.zeros_like(thresholds, dtype=np.float64)
    frr = np.zeros_like(thresholds, dtype=np.float64)
    n_g = max(1, len(scores_genuine))
    n_f = max(1, len(scores_forgery))
    for i, t in enumerate(thresholds):
        far[i] = (scores_forgery >= t).sum() / n_f
        frr[i] = (scores_genuine < t).sum() / n_g
    return far, frr


def _trapezoid_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """AUC por regla del trapecio. Asume fpr ordenado ascendente."""
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def evaluate(
    scores_genuine: np.ndarray,
    scores_forgery: np.ndarray,
    n_thresholds: int = 1001,
) -> EvalMetrics:
    """Calcula FAR/FRR/EER/ROC/AUC barriendo umbrales."""
    sg = np.asarray(scores_genuine, dtype=np.float64)
    sf = np.asarray(scores_forgery, dtype=np.float64)

    lo = float(min(sg.min(), sf.min())) if len(sg) and len(sf) else 0.0
    hi = float(max(sg.max(), sf.max())) if len(sg) and len(sf) else 1.0
    thresholds = np.linspace(lo, hi, n_thresholds)

    far, frr = _compute_far_frr(sg, sf, thresholds)

    # EER = punto donde |FAR - FRR| es mínimo.
    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    tpr = 1.0 - frr
    fpr = far
    auc = _trapezoid_auc(fpr, tpr)

    return EvalMetrics(
        thresholds=thresholds,
        far=far,
        frr=frr,
        tpr=tpr,
        fpr=fpr,
        eer=eer,
        eer_threshold=eer_threshold,
        auc=auc,
        n_genuine=len(sg),
        n_forgery=len(sf),
    )


def plot_roc(metrics: EvalMetrics, save_to: str | Path | None = None) -> None:
    """Plotea curva ROC. Requiere matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(metrics.fpr)
    ax.plot(metrics.fpr[order], metrics.tpr[order], lw=2,
            label=f"AUC = {metrics.auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.scatter(
        [metrics.far[np.argmin(np.abs(metrics.far - metrics.frr))]],
        [1 - metrics.frr[np.argmin(np.abs(metrics.far - metrics.frr))]],
        c="red", s=50, zorder=5,
        label=f"EER = {metrics.eer:.3f}",
    )
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title("Curva ROC del verificador de firmas")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_far_frr(metrics: EvalMetrics, save_to: str | Path | None = None) -> None:
    """Plotea FAR y FRR vs umbral. El cruce visualiza el EER."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(metrics.thresholds, metrics.far, label="FAR", lw=2)
    ax.plot(metrics.thresholds, metrics.frr, label="FRR", lw=2)
    ax.axvline(
        metrics.eer_threshold, color="red", linestyle="--",
        label=f"EER threshold = {metrics.eer_threshold:.3f}",
    )
    ax.axhline(metrics.eer, color="red", linestyle=":", alpha=0.5,
               label=f"EER = {metrics.eer:.3f}")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Tasa de error")
    ax.set_title("FAR vs FRR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_score_histograms(
    scores_genuine: np.ndarray,
    scores_forgery: np.ndarray,
    save_to: str | Path | None = None,
) -> None:
    """Histograma comparado de scores genuinos vs falsificaciones."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores_genuine, bins=bins, alpha=0.6, label="Genuinos", density=True)
    ax.hist(scores_forgery, bins=bins, alpha=0.6, label="Falsificaciones", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de scores: genuinos vs falsificaciones")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def summary(metrics: EvalMetrics) -> str:
    """String con el resumen de las métricas, listo para imprimir."""
    return (
        f"Pares evaluados:    {metrics.n_genuine} genuinos, {metrics.n_forgery} falsificaciones\n"
        f"AUC:                {metrics.auc:.4f}\n"
        f"EER:                {metrics.eer:.4f}  (FAR=FRR={metrics.eer*100:.2f}%)\n"
        f"Umbral óptimo:      {metrics.eer_threshold:.4f}"
    )
