"""W1K step 2: STTC code *families* (record-level unions) and filter sensitivity.

Reuses the header-only Dx cache built by W1K_scan.py (processed root, w1k_dx_cache.json).
Appends family rows to W1K_sttc_code_histogram.csv and writes W1K_filter_sensitivity.csv.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
from igraphecg.config import EXCLUDE_SUPERCLASS, SUPERCLASSES
from igraphecg.labels.map_snomed import DEFAULT_MAP_CSV, load_snomed_map

OUT = repro.outdir("runs/v7rev_stats")
CACHE = repro.processed() / "w1k_dx_cache.json"
if not CACHE.exists():
    raise SystemExit(f"no Dx cache at {CACHE}: run W1K_scan.py first")
DX = json.loads(CACHE.read_text(encoding="utf-8"))
SNOMED = load_snomed_map(DEFAULT_MAP_CSV)
TARGETS = set(SUPERCLASSES)

FAMILIES = {
    "FAMILY_T_wave":            ["164934002", "59931005"],                  # T abnormal, T inversion
    "FAMILY_ST_elevation":      ["164930006"],
    "FAMILY_ST_depression":     ["429622005"],
    "FAMILY_ST_T_nonspecific":  ["428750005", "164931005"],                 # nonspec ST-T, ST-T change
    "FAMILY_ischemia":          ["413444003"],
}

fam_rows, sens_rows = [], []
for cohort, recs in DX.items():
    n = len(recs)
    fam_all = Counter(); fam_surv = Counter()
    n_any_all = 0; n_surv = 0
    strict = Counter(); relaxed_other = Counter(); anycode = Counter()
    for r in recs:
        u = sorted({str(c).strip() for c in r["dx"] if str(c).strip()})
        mapped = {SNOMED.get(c) for c in u}; mapped.discard(None)
        tgt = mapped & TARGETS
        has_hyp = EXCLUDE_SUPERCLASS in mapped
        others = mapped - TARGETS - {EXCLUDE_SUPERCLASS}
        y_strict = (next(iter(tgt)) if (not has_hyp and not others and len(tgt) == 1) else None)
        y_relax = (next(iter(tgt)) if (not has_hyp and len(tgt) == 1) else None)   # OTHER tolerated
        if y_strict:
            strict[y_strict] += 1
        if y_relax:
            relaxed_other[y_relax] += 1
        for t in tgt:                       # "carries the class at all", no exclusivity
            anycode[t] += 1
        s_codes = [c for c in u if SNOMED.get(c) == "STTC"]
        if s_codes:
            n_any_all += 1
        if y_strict == "STTC":
            n_surv += 1
        for fam, codes in FAMILIES.items():
            if any(c in u for c in codes):
                fam_all[fam] += 1
                if y_strict == "STTC":
                    fam_surv[fam] += 1
    for fam in FAMILIES:
        fam_rows.append({
            "cohort": cohort, "snomed_code": fam,
            "term": "+".join(FAMILIES[fam]),
            "n_records_with_code_all": fam_all.get(fam, 0),
            "pct_of_records_with_any_STTC_code": round(100.0 * fam_all.get(fam, 0) / n_any_all, 2) if n_any_all else 0.0,
            "n_records_with_code_among_clean_STTC": fam_surv.get(fam, 0),
            "pct_of_clean_STTC_records": round(100.0 * fam_surv.get(fam, 0) / n_surv, 2) if n_surv else 0.0,
        })
    def retention(strict_n, any_n):
        """strict survivors as a percentage of records carrying the class at all; blank when none do."""
        return round(100.0 * strict_n / any_n, 2) if any_n else ""

    for cls in SUPERCLASSES:
        sens_rows.append({
            "cohort": cohort, "superclass": cls,
            "n_records_carrying_class_any": anycode.get(cls, 0),
            "n_kept_relaxed_OTHER_tolerated": relaxed_other.get(cls, 0),
            "n_kept_strict_paper_filter": strict.get(cls, 0),
            "pct_of_cohort_strict": round(100.0 * strict.get(cls, 0) / n, 2),
            "retention_pct_strict_over_any": retention(strict.get(cls, 0), anycode.get(cls, 0)),
        })
    sens_rows.append({
        "cohort": cohort, "superclass": "TOTAL",
        "n_records_carrying_class_any": sum(anycode.values()),
        "n_kept_relaxed_OTHER_tolerated": sum(relaxed_other.values()),
        "n_kept_strict_paper_filter": sum(strict.values()),
        "pct_of_cohort_strict": round(100.0 * sum(strict.values()) / n, 2),
        "retention_pct_strict_over_any": retention(sum(strict.values()), sum(anycode.values())),
    })

hist = OUT / "W1K_sttc_code_histogram.csv"
rows = list(csv.DictReader(open(hist, newline="", encoding="utf-8")))
flds = list(rows[0].keys())
rows = [r for r in rows if not r["snomed_code"].startswith("FAMILY")]   # idempotent re-run
rows += [{k: r.get(k, "") for k in flds} for r in fam_rows]
rows.sort(key=lambda r: (r["cohort"], r["snomed_code"].startswith("FAMILY"),
                         r["snomed_code"] == "ANY_STTC_CODE", r["snomed_code"]))
with open(hist, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=flds); w.writeheader(); w.writerows(rows)

with open(OUT / "W1K_filter_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cohort", "superclass", "n_records_carrying_class_any",
                                      "n_kept_relaxed_OTHER_tolerated",
                                      "n_kept_strict_paper_filter", "pct_of_cohort_strict",
                                      "retention_pct_strict_over_any"])
    w.writeheader(); w.writerows(sens_rows)

print(open(hist, encoding="utf-8").read())
print(open(OUT / "W1K_filter_sensitivity.csv", encoding="utf-8").read())
