"""W1-E: PTB-XL inclusion accounting and MI case-mix, for reviewer R2 comment 1.

Reproduces the clean-subset filter with the repository's own label machinery, then
quantifies (a) how the filter reshapes the class prior and (b) whether it shifts the
MI subclass mix. The subclass shift is computed under several counting conventions
because the adversarial check showed the absolute percentages flip between them;
only statements stable across all conventions may be published.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
from igraphecg.utils.paths import find_ptbxl_root
RAW = find_ptbxl_root()          # was an absolute path on the author's machine
if RAW is None:
    raise SystemExit("PTB-XL not found: set ECG_DATA_DIR to the directory holding "
                     "ptbxl_database.csv (see data/README.md)")
OUT = repro.outdir("runs/v7rev_stats")

from igraphecg.data.label_utils import (TARGET_CLASSES, build_code_to_superclass,
                                        parse_scp_codes, record_superclasses)

db = pd.read_csv(RAW / "ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(RAW / "scp_statements.csv", index_col=0)
code2super = build_code_to_superclass(scp)
diag = scp[scp.get("diagnostic") == 1] if "diagnostic" in scp.columns else scp
code2sub = {str(c): r.get("diagnostic_subclass") for c, r in diag.iterrows()
            if isinstance(r.get("diagnostic_subclass"), str)}

TARGET = set(TARGET_CLASSES)
print("=" * 78)
print("W1-E  PTB-XL inclusion accounting")
print("=" * 78)
print(f"total records in ptbxl_database.csv: {len(db)}")

rows = []
for ecg_id, r in db.iterrows():
    codes = parse_scp_codes(r.get("scp_codes"))
    supers = record_superclasses(codes, code2super, 0.0)
    tgt = supers & TARGET
    rows.append({"ecg_id": ecg_id, "fold": int(r["strat_fold"]),
                 "n_target": len(tgt), "has_hyp": "HYP" in supers,
                 "targets": "|".join(sorted(tgt)), "n_super": len(supers)})
M = pd.DataFrame(rows).set_index("ecg_id")

no_super = (M.n_super == 0).sum()
has_super = (M.n_super > 0).sum()
no_target = (M.n_target == 0).sum()
multi = (M.n_target > 1).sum()
single = (M.n_target == 1).sum()
single_hyp = ((M.n_target == 1) & M.has_hyp).sum()
clean = ((M.n_target == 1) & ~M.has_hyp).sum()

print(f"  no diagnostic superclass at all : {no_super}")
print(f"  >=1 diagnostic superclass       : {has_super}")
print(f"  zero of the four target classes : {no_target}")
print(f"  exactly one target class        : {single}")
print(f"     of which carry HYP overlap   : {single_hyp}   (excluded)")
print(f"  >1 target class (multi-label)   : {multi}   (excluded)")
print(f"  CLEAN SUBSET                    : {clean}")
print(f"  paper states 15,709 -> {'MATCH' if clean == 15709 else 'MISMATCH'}")

# --- retention by class ---
print("\n--- retention by target class (records carrying that class at all) ---")
ret = []
for c in TARGET_CLASSES:
    carries = M.targets.str.split("|").apply(lambda s: c in s)
    kept = carries & (M.n_target == 1) & ~M.has_hyp
    ret.append({"class": c, "carrying": int(carries.sum()), "retained": int(kept.sum()),
                "retention_pct": 100 * kept.sum() / carries.sum()})
    print(f"  {c:5s} carried by {carries.sum():6d}  retained {kept.sum():6d}  "
          f"= {100*kept.sum()/carries.sum():5.1f}%")
retdf = pd.DataFrame(ret)
n_norm, n_mi = retdf.set_index("class").loc["NORM"], retdf.set_index("class").loc["MI"]
print(f"\n  NORM:MI ratio  before filter {n_norm.carrying/n_mi.carrying:.2f}"
      f"   after filter {n_norm.retained/n_mi.retained:.2f}")

# --- class prior, full vs clean (feeds the prior-corrected argmax in W1-D) ---
print("\n--- class prior over records with >=1 target class ---")
pri = []
for c in TARGET_CLASSES:
    carries = M.targets.str.split("|").apply(lambda s: c in s)
    pri.append(carries.sum())
pri = np.array(pri, dtype=float)
print("  full  (>=1 target, records may count in several classes):",
      np.round(pri / pri.sum(), 4).tolist())
cleanmask = (M.n_target == 1) & ~M.has_hyp
cl = np.array([(M.loc[cleanmask, "targets"] == c).sum() for c in TARGET_CLASSES], dtype=float)
print("  clean (single-label, HYP-free)                        :",
      np.round(cl / cl.sum(), 4).tolist())
print("  clean counts:", dict(zip(TARGET_CLASSES, cl.astype(int).tolist())))
np.save(OUT / "w1e_prior_full.npy", pri / pri.sum())

# --- MI subclass mix under several counting conventions ---
print("\n" + "-" * 78)
print("MI subclass mix: retained vs excluded, under four counting conventions")
print("-" * 78)
MI_SUBS = ["AMI", "IMI", "LMI", "PMI"]
carries_mi = M.targets.str.split("|").apply(lambda s: "MI" in s)
kept_mi = carries_mi & cleanmask
excl_mi = carries_mi & ~cleanmask

def subclass_share(mask, dedup, min_like):
    cnt = Counter()
    for ecg_id in M.index[mask]:
        codes = parse_scp_codes(db.loc[ecg_id, "scp_codes"])
        subs = [code2sub.get(c) for c, v in codes.items() if v >= min_like]
        subs = [s for s in subs if s in MI_SUBS]
        for s in (set(subs) if dedup else subs):
            cnt[s] += 1
    tot = sum(cnt.values())
    return {s: 100 * cnt.get(s, 0) / tot for s in MI_SUBS}, tot

conventions = [("dedup within record, all codes", True, 0.0),
               ("count every code, all codes", False, 0.0),
               ("dedup, likelihood>=50", True, 50.0),
               ("dedup, likelihood==100", True, 100.0)]
recs = []
for name, dedup, ml in conventions:
    k, kt = subclass_share(kept_mi, dedup, ml)
    e, et = subclass_share(excl_mi, dedup, ml)
    print(f"\n  [{name}]  retained n={kt}, excluded n={et}")
    print("    retained :", "  ".join(f"{s} {k[s]:5.1f}%" for s in MI_SUBS))
    print("    excluded :", "  ".join(f"{s} {e[s]:5.1f}%" for s in MI_SUBS))
    print(f"    IMI share higher in retained? {k['IMI'] > e['IMI']}"
          f"   AMI share lower in retained? {k['AMI'] < e['AMI']}")
    for s in MI_SUBS:
        recs.append({"convention": name, "subclass": s, "retained_pct": k[s],
                     "excluded_pct": e[s], "n_retained_tags": kt, "n_excluded_tags": et})
sub = pd.DataFrame(recs)
sub.to_csv(OUT / "w1e_mi_subclass_by_convention.csv", index=False)

print("\n" + "-" * 78)
imi = sub[sub.subclass == "IMI"]
ami = sub[sub.subclass == "AMI"]
print("STABLE ACROSS ALL CONVENTIONS?")
print(f"  retained IMI share > excluded IMI share : {bool((imi.retained_pct > imi.excluded_pct).all())}")
print(f"  retained AMI share < excluded AMI share : {bool((ami.retained_pct < ami.excluded_pct).all())}")
print(f"  retained pool is IMI-dominant (IMI>AMI) : "
      f"{[bool(x) for x in (imi.retained_pct.values > ami.retained_pct.values)]}  <- NOT stable")

M.to_csv(OUT / "w1e_record_disposition.csv")
retdf.to_csv(OUT / "w1e_retention_by_class.csv", index=False)
print(f"\n-> {OUT}")
