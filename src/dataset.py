"""Helpers para cargar CEDAR y construir pares de evaluación.

Estructura esperada en data/:
    data/genuine/original_<person>_<n>.png
    data/skilled_forgery/forgeries_<person>_<n>.png

Donde <person> va de 1 a 55 y <n> de 1 a 24.

Genera tres tipos de pares para evaluación:
    - genuine pairs:         (firma_i, firma_j) de la MISMA persona, ambas genuinas
    - skilled_forgery pairs: (genuina_i, falsificacion_j) de la MISMA persona
    - random_forgery pairs:  (genuina_personaA, genuina_personaB) personas distintas
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GENUINE_DIR = "genuine"
FORGERY_DIR = "skilled_forgery"

# Patrones flexibles para soportar variantes de naming en CEDAR.
GENUINE_PATTERNS = (
    re.compile(r"original_(\d+)_(\d+)\.(?:png|PNG|jpg|JPG)$"),
    re.compile(r"genuine_(\d+)_(\d+)\.(?:png|PNG|jpg|JPG)$"),
)
FORGERY_PATTERNS = (
    re.compile(r"forgeries_(\d+)_(\d+)\.(?:png|PNG|jpg|JPG)$"),
    re.compile(r"forgery_(\d+)_(\d+)\.(?:png|PNG|jpg|JPG)$"),
)


@dataclass
class SignatureRecord:
    path: Path
    person_id: int
    sample_id: int
    is_genuine: bool


@dataclass
class Pair:
    a: Path
    b: Path
    label: str   # "genuine" | "skilled_forgery" | "random_forgery"


def _scan_directory(
    directory: Path, patterns: tuple[re.Pattern, ...], is_genuine: bool
) -> list[SignatureRecord]:
    records: list[SignatureRecord] = []
    if not directory.exists():
        return records
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        for pat in patterns:
            m = pat.match(path.name)
            if m:
                records.append(
                    SignatureRecord(
                        path=path,
                        person_id=int(m.group(1)),
                        sample_id=int(m.group(2)),
                        is_genuine=is_genuine,
                    )
                )
                break
    return records


def load_cedar(data_root: str | Path) -> tuple[list[SignatureRecord], list[SignatureRecord]]:
    """Devuelve (genuinas, falsificaciones_habiles) escaneados desde data_root."""
    root = Path(data_root)
    genuines = _scan_directory(root / GENUINE_DIR, GENUINE_PATTERNS, True)
    forgeries = _scan_directory(root / FORGERY_DIR, FORGERY_PATTERNS, False)
    return genuines, forgeries


def _group_by_person(records: list[SignatureRecord]) -> dict[int, list[SignatureRecord]]:
    grouped: dict[int, list[SignatureRecord]] = defaultdict(list)
    for r in records:
        grouped[r.person_id].append(r)
    return dict(grouped)


def build_pairs(
    genuines: list[SignatureRecord],
    forgeries: list[SignatureRecord],
    n_per_category: int = 1500,
    seed: int = 42,
) -> list[Pair]:
    """Construye una muestra balanceada de pares para evaluación.

    n_per_category controla el tamaño por categoría (genuine, skilled, random).
    """
    rng = np.random.default_rng(seed)
    g_by_p = _group_by_person(genuines)
    f_by_p = _group_by_person(forgeries)
    persons = sorted(g_by_p.keys())

    pairs: list[Pair] = []

    # 1. Pares genuinos: combinaciones dentro de la misma persona.
    genuine_pairs: list[Pair] = []
    for p, samples in g_by_p.items():
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                genuine_pairs.append(
                    Pair(samples[i].path, samples[j].path, "genuine")
                )
    rng.shuffle(genuine_pairs)
    pairs.extend(genuine_pairs[:n_per_category])

    # 2. Pares de falsificación hábil: genuina + falsificación de la misma persona.
    skilled_pairs: list[Pair] = []
    for p in persons:
        if p not in f_by_p:
            continue
        for g in g_by_p[p]:
            for f in f_by_p[p]:
                skilled_pairs.append(Pair(g.path, f.path, "skilled_forgery"))
    rng.shuffle(skilled_pairs)
    pairs.extend(skilled_pairs[:n_per_category])

    # 3. Pares de falsificación aleatoria: dos genuinas de personas distintas.
    random_pairs: list[Pair] = []
    if len(persons) >= 2:
        attempts = 0
        target = n_per_category
        while len(random_pairs) < target and attempts < target * 20:
            attempts += 1
            pa, pb = rng.choice(persons, size=2, replace=False)
            sa = g_by_p[int(pa)][rng.integers(0, len(g_by_p[int(pa)]))]
            sb = g_by_p[int(pb)][rng.integers(0, len(g_by_p[int(pb)]))]
            random_pairs.append(Pair(sa.path, sb.path, "random_forgery"))
    pairs.extend(random_pairs)

    return pairs


def split_train_test(
    pairs: list[Pair], train_ratio: float = 0.3, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Split estratificado por label."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_label[p.label].append(p)

    train: list[Pair] = []
    test: list[Pair] = []
    for label, group in by_label.items():
        idx = np.arange(len(group))
        rng.shuffle(idx)
        cut = int(len(group) * train_ratio)
        train.extend([group[i] for i in idx[:cut]])
        test.extend([group[i] for i in idx[cut:]])
    return train, test
