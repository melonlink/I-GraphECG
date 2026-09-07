"""Round 3 shared statistics: Kruskal-Wallis + BH-FDR + Cliff's delta, plus a per-class table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata

from ..data.label_utils import TARGET_CLASSES


def bh_fdr(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = P(a>b)-P(a<b), rank-based implementation, O((n+m)log)."""
    a = np.asarray(a); b = np.asarray(b); na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    # Average ranks are required for ties. Ordinal ranks bias delta whenever a
    # descriptor is discrete or clipped, which is common for ECG morphology.
    allv = np.concatenate([a, b]); ranks = rankdata(allv, method="average")
    u = ranks[:na].sum() - na * (na + 1) / 2
    return float(2 * u / (na * nb) - 1)


def by_class_table(feats: dict[str, np.ndarray], labels: np.ndarray,
                   expect: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Per-feature class mean±std, KW test, BH-FDR; optional direction check (disease vs NORM).

    expect: {feature: (disease_class, 'up'|'down')}
    """
    expect = expect or {}
    norm_i = TARGET_CLASSES.index("NORM")
    rows, pvals = [], []
    for fname, vals in feats.items():
        vals = np.asarray(vals)
        groups = [vals[labels == c] for c in range(len(TARGET_CLASSES))]
        try:
            _, p = kruskal(*[g for g in groups if len(g) > 0])
        except ValueError:
            p = 1.0
        pvals.append(p)
        row = {"feature": fname, "kw_p": p}
        for c, cn in enumerate(TARGET_CLASSES):
            g = vals[labels == c]
            row[f"{cn}_mean"] = float(np.mean(g)) if len(g) else float("nan")
            row[f"{cn}_std"] = float(np.std(g)) if len(g) else float("nan")
        if fname in expect:
            dis, direction = expect[fname]
            delta = cliffs_delta(vals[labels == TARGET_CLASSES.index(dis)], vals[labels == norm_i])
            row["expect"] = f"{dis} {direction} vs NORM"
            row["cliffs_delta"] = delta
            row["direction_ok"] = bool((delta > 0) if direction == "up" else (delta < 0))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["fdr_q"] = bh_fdr([p if np.isfinite(p) else 1.0 for p in pvals])
    return df


def cliffs_vs_norm(vals: np.ndarray, labels: np.ndarray, dis: str) -> float:
    norm_i = TARGET_CLASSES.index("NORM")
    return cliffs_delta(np.asarray(vals)[labels == TARGET_CLASSES.index(dis)],
                        np.asarray(vals)[labels == norm_i])
