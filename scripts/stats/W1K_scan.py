"""W1K step 1: header-only SNOMED scan of Georgia (G12EC) and CPSC2018.

Reads ONLY .hea files (wfdb.rdheader) -- no signal I/O. Reuses the repository's frozen
label machinery: igraphecg.labels.map_snomed.load_snomed_map / map_record_label and
igraphecg.labels.superclass.clean_single_label, plus the WFDB adapter's _parse_dx.

Data    PhysioNet/CinC Challenge 2021 `training/georgia` and `training/cpsc_2018`, found
        under --challenge_dir (default: <data root>/../challenge_2021/training, the layout
        the notes describe next to ECG_DATA_DIR). The parsed Dx lists are cached under the
        processed root (w1k_dx_cache.json) so the scan runs once.
Writes  runs/v7rev_stats/W1K_controls.csv            expected cohort counts, PASS/FAIL
        runs/v7rev_stats/W1K_cohort_disposition.csv  where every record goes under the clean filter
        runs/v7rev_stats/W1K_mi_attrition.csv        why no MI record survives
        runs/v7rev_stats/W1K_mi_records_detail.csv
        runs/v7rev_stats/W1K_sttc_code_histogram.csv (extended by W1K_addendum.py)
        runs/v7rev_stats/W1K_full_code_histogram.json
Then run W1K_addendum.py.

Written for the reviewer response (cohort label composition, Supplementary); it ran from a
scratch directory with absolute paths and is recorded here resolving through igraphecg.repro.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
from igraphecg.config import EXCLUDE_SUPERCLASS, SUPERCLASSES
from igraphecg.labels.map_snomed import DEFAULT_MAP_CSV, load_snomed_map, map_record_label
from igraphecg.labels.superclass import clean_single_label
from igraphecg.sources.wfdb_source import _parse_dx

OUT = repro.outdir("runs/v7rev_stats")
CACHE = repro.processed() / "w1k_dx_cache.json"


def challenge_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("ECG_CHALLENGE_DIR")
    return Path(env) if env else repro.data().parent / "challenge_2021" / "training"


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--challenge_dir", default=None,
                help="directory holding georgia/ and cpsc_2018/ (default: ECG_CHALLENGE_DIR or "
                     "<data root>/../challenge_2021/training); not needed if the Dx cache exists")
args = ap.parse_args()
CH = challenge_dir(args.challenge_dir)
COHORTS = {"georgia": CH / "georgia", "cpsc_2018": CH / "cpsc_2018"}

SNOMED = load_snomed_map(DEFAULT_MAP_CSV)
TERMS = {}
with open(DEFAULT_MAP_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = str(row["snomed_code"]).strip()
        if c and not c.startswith("#"):
            TERMS[c] = str(row["term"]).strip()
TARGETS = set(SUPERCLASSES)

_CACHE = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def scan(name, root):
    if name in _CACHE:
        return _CACHE[name]
    if not Path(root).exists():
        raise SystemExit(f"{name}: no Dx cache at {CACHE} and no headers under {root}; "
                         "pass --challenge_dir or set ECG_CHALLENGE_DIR")
    import wfdb
    paths = sorted(p.with_suffix("") for p in Path(root).rglob("*.hea"))
    out = []
    for stem in paths:
        hdr = wfdb.rdheader(str(stem))
        out.append({"id": stem.name, "dx": _parse_dx(getattr(hdr, "comments", []) or [])})
    _CACHE[name] = out
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(_CACHE), encoding="utf-8")
    return out


def analyze(recs):
    code_hist = Counter()
    unmapped = Counter()
    disp = Counter()
    kept_by_class = Counter()
    mi_rows = []
    sttc_all = Counter()
    sttc_surv = Counter()
    sttc_pairs = Counter()
    n_any_sttc_code = 0
    n_any_sttc_code_surv = 0

    for r in recs:
        ucodes = sorted({str(c).strip() for c in r["dx"] if str(c).strip()})
        for c in ucodes:
            code_hist[c] += 1
            if c not in SNOMED:
                unmapped[c] += 1
        mapped = {SNOMED.get(c) for c in ucodes}
        mapped.discard(None)
        tgt = mapped & TARGETS
        has_hyp = EXCLUDE_SUPERCLASS in mapped
        others = mapped - TARGETS - {EXCLUDE_SUPERCLASS}

        y = map_record_label(r["dx"], SNOMED)          # repository function (authoritative)
        assert y == clean_single_label(mapped), (r["id"], y)

        disp["n_records"] += 1
        if not ucodes:
            disp["n_no_dx_at_all"] += 1
        if any(c not in SNOMED for c in ucodes):
            disp["n_with_unmapped_code"] += 1
        if len(tgt) >= 1:
            disp["n_ge1_target_superclass"] += 1
        if len(tgt) == 0:
            disp["n_zero_target_superclass"] += 1
        if len(tgt) == 1:
            disp["n_exactly1_target_superclass"] += 1
        if len(tgt) >= 2:
            disp["n_multi_target_superclass"] += 1
        if has_hyp:
            disp["n_carrying_HYP"] += 1
        if others:
            disp["n_carrying_OTHER_code"] += 1

        # mutually exclusive attrition, in clean_single_label's own precedence order
        if has_hyp:
            disp["drop_1_HYP_overlap"] += 1
        elif others:
            disp["drop_2_OTHER_code_present"] += 1
        elif len(tgt) >= 2:
            disp["drop_3_multi_superclass"] += 1
        elif len(tgt) == 0:
            disp["drop_4_no_target_superclass"] += 1
        else:
            disp["n_clean_single_label_kept"] += 1
            kept_by_class[y] += 1

        mi_codes = [c for c in ucodes if SNOMED.get(c) == "MI"]
        if mi_codes:
            if has_hyp:
                reason = "excluded_HYP_overlap"
            elif others:
                reason = "excluded_OTHER_code_present"
            elif len(tgt) >= 2:
                reason = "excluded_multi_superclass"
            else:
                reason = "kept_MI"
            mi_rows.append({
                "record_id": r["id"], "mi_codes": "|".join(mi_codes),
                "all_codes": "|".join(ucodes),
                "all_terms": "|".join(TERMS.get(c, "UNMAPPED") for c in ucodes),
                "n_target_superclasses": len(tgt),
                "target_superclasses": "|".join(sorted(tgt)),
                "has_HYP": int(has_hyp), "has_OTHER": int(bool(others)),
                "disposition": reason,
            })

        s_codes = [c for c in ucodes if SNOMED.get(c) == "STTC"]
        if s_codes:
            n_any_sttc_code += 1
        for c in s_codes:
            sttc_all[c] += 1
            if y == "STTC":
                sttc_surv[c] += 1
        if y == "STTC":
            n_any_sttc_code_surv += 1
            sttc_pairs["+".join(sorted(s_codes))] += 1

    disp["n_records_with_any_STTC_code"] = n_any_sttc_code
    return dict(disp=disp, kept_by_class=kept_by_class, code_hist=code_hist,
                unmapped=unmapped, mi_rows=mi_rows, sttc_all=sttc_all,
                sttc_surv=sttc_surv, sttc_pairs=sttc_pairs,
                n_any_sttc=n_any_sttc_code, n_any_sttc_surv=n_any_sttc_code_surv)


res = {}
for name, root in COHORTS.items():
    print("scanning %s ..." % name, flush=True)
    res[name] = analyze(scan(name, root))
    d = res[name]["disp"]
    print("  %s: %d records, kept %d, by class %s"
          % (name, d["n_records"], d["n_clean_single_label_kept"],
             dict(res[name]["kept_by_class"])), flush=True)

# ---------------- CONTROLS: the cohort counts the paper reports ----------------
CONTROLS = {
    "georgia":   {"total": 10344, "kept": 2586, "NORM": 1752, "STTC": 542, "CD": 292, "MI": 0},
    "cpsc_2018": {"total": 6877,  "kept": 4305, "NORM": 918,  "STTC": 971, "CD": 2416, "MI": 0},
}
ctrl_report = []
ok = True
for name, c in CONTROLS.items():
    d = res[name]["disp"]
    k = res[name]["kept_by_class"]
    checks = [("total_records", d["n_records"], c["total"]),
              ("clean_kept", d["n_clean_single_label_kept"], c["kept"])]
    for cls in ("NORM", "MI", "STTC", "CD"):
        checks.append(("kept_" + cls, k.get(cls, 0), c[cls]))
    for what, got, exp in checks:
        good = (got == exp)
        ok = ok and good
        ctrl_report.append({"cohort": name, "quantity": what, "got": got,
                            "expected": exp, "match": "PASS" if good else "FAIL"})
with open(OUT / "W1K_controls.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cohort", "quantity", "got", "expected", "match"])
    w.writeheader(); w.writerows(ctrl_report)
print("CONTROLS:", "ALL PASS" if ok else "FAILURE")
for r in ctrl_report:
    print("  %-10s %-14s got=%6d exp=%6d %s"
          % (r["cohort"], r["quantity"], r["got"], r["expected"], r["match"]))

# ---------------- 1. disposition ----------------
KEYS = ["n_records", "n_no_dx_at_all", "n_with_unmapped_code",
        "n_ge1_target_superclass", "n_zero_target_superclass",
        "n_exactly1_target_superclass", "n_multi_target_superclass",
        "n_carrying_HYP", "n_carrying_OTHER_code", "n_records_with_any_STTC_code",
        "drop_1_HYP_overlap", "drop_2_OTHER_code_present",
        "drop_3_multi_superclass", "drop_4_no_target_superclass",
        "n_clean_single_label_kept"]
rows = []
for name in COHORTS:
    d = res[name]["disp"]; k = res[name]["kept_by_class"]; n = d["n_records"]
    for key in KEYS:
        rows.append({"cohort": name, "quantity": key, "n_records": d.get(key, 0),
                     "pct_of_cohort": round(100.0 * d.get(key, 0) / n, 2)})
    for cls in ("NORM", "MI", "STTC", "CD"):
        rows.append({"cohort": name, "quantity": "kept_" + cls, "n_records": k.get(cls, 0),
                     "pct_of_cohort": round(100.0 * k.get(cls, 0) / n, 2)})
with open(OUT / "W1K_cohort_disposition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cohort", "quantity", "n_records", "pct_of_cohort"])
    w.writeheader(); w.writerows(rows)

# ---------------- 2. MI attrition ----------------
mi_rows = []
for name in COHORTS:
    mr = res[name]["mi_rows"]
    tot = len(mr)
    cnt = Counter(r["disposition"] for r in mr)
    co_hyp = sum(r["has_HYP"] for r in mr)
    co_other = sum(r["has_OTHER"] for r in mr)
    co_multi = sum(1 for r in mr if r["n_target_superclasses"] >= 2)
    co_class = Counter()
    for r in mr:
        for t in r["target_superclasses"].split("|"):
            if t and t != "MI":
                co_class[t] += 1
    n = res[name]["disp"]["n_records"]

    def add(q, v, note=""):
        mi_rows.append({"cohort": name, "quantity": q, "n_records": v,
                        "pct_of_MI_coded": (round(100.0 * v / tot, 2) if tot else ""),
                        "pct_of_cohort": round(100.0 * v / n, 3), "note": note})

    add("n_records_with_any_MI_mapped_code", tot,
        "164865005 'myocardial infarction' is the ONLY MI-mapped SNOMED code in the study map")
    add("excluded_HYP_overlap", cnt.get("excluded_HYP_overlap", 0), "attrition precedence 1")
    add("excluded_OTHER_code_present", cnt.get("excluded_OTHER_code_present", 0),
        "attrition precedence 2: co-occurring rhythm/axis/etc code maps to OTHER")
    add("excluded_multi_superclass", cnt.get("excluded_multi_superclass", 0),
        "attrition precedence 3: MI plus another of the four superclasses")
    add("kept_clean_single_label_MI", cnt.get("kept_MI", 0), "survives the clean filter")
    add("overlap_any_HYP", co_hyp, "non-exclusive overlap count")
    add("overlap_any_OTHER", co_other, "non-exclusive overlap count")
    add("overlap_any_second_target_superclass", co_multi, "non-exclusive overlap count")
    for cls, v in sorted(co_class.items()):
        add("co_occurring_superclass_" + cls, v, "non-exclusive overlap count")
with open(OUT / "W1K_mi_attrition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cohort", "quantity", "n_records",
                                      "pct_of_MI_coded", "pct_of_cohort", "note"])
    w.writeheader(); w.writerows(mi_rows)

allmi = []
for name in COHORTS:
    for r in res[name]["mi_rows"]:
        allmi.append(dict([("cohort", name)] + list(r.items())))
FIELDS = ["cohort", "record_id", "mi_codes", "all_codes", "all_terms",
          "n_target_superclasses", "target_superclasses", "has_HYP", "has_OTHER", "disposition"]
with open(OUT / "W1K_mi_records_detail.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(allmi)

# ---------------- 3. STTC code histogram ----------------
sttc_codes = sorted([c for c, m in SNOMED.items() if m == "STTC"])
srows = []
for name in COHORTS:
    a = res[name]["sttc_all"]; s = res[name]["sttc_surv"]
    n_any = res[name]["n_any_sttc"]
    n_surv = res[name]["kept_by_class"].get("STTC", 0)
    for c in sttc_codes:
        srows.append({
            "cohort": name, "snomed_code": c, "term": TERMS.get(c, ""),
            "n_records_with_code_all": a.get(c, 0),
            "pct_of_records_with_any_STTC_code": (round(100.0 * a.get(c, 0) / n_any, 2) if n_any else 0.0),
            "n_records_with_code_among_clean_STTC": s.get(c, 0),
            "pct_of_clean_STTC_records": (round(100.0 * s.get(c, 0) / n_surv, 2) if n_surv else 0.0),
        })
    srows.append({"cohort": name, "snomed_code": "ANY_STTC_CODE", "term": "(union, record-level)",
                  "n_records_with_code_all": n_any,
                  "pct_of_records_with_any_STTC_code": 100.0,
                  "n_records_with_code_among_clean_STTC": n_surv,
                  "pct_of_clean_STTC_records": 100.0 if n_surv else 0.0})
with open(OUT / "W1K_sttc_code_histogram.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cohort", "snomed_code", "term",
                                      "n_records_with_code_all",
                                      "pct_of_records_with_any_STTC_code",
                                      "n_records_with_code_among_clean_STTC",
                                      "pct_of_clean_STTC_records"])
    w.writeheader(); w.writerows(srows)

# ---------------- supporting JSON ----------------
dump = {}
for name in COHORTS:
    r = res[name]
    dump[name] = {
        "n_records": r["disp"]["n_records"],
        "disposition": dict(r["disp"]),
        "kept_by_class": dict(r["kept_by_class"]),
        "unmapped_codes": dict(r["unmapped"]),
        "sttc_code_combination_among_clean_STTC": dict(r["sttc_pairs"].most_common()),
        "code_histogram": {c: {"n": v, "term": TERMS.get(c, "UNMAPPED"),
                               "mapped": SNOMED.get(c, "UNMAPPED")}
                           for c, v in r["code_hist"].most_common()},
    }
dump["_meta"] = {
    # relative to the tree root: a registered file must not carry the machine it was made on
    "label_map": _Path(DEFAULT_MAP_CSV).resolve().relative_to(repro.repo()).as_posix(),
    "controls_all_pass": bool(ok),
    "header_only_scan": True,
    "n_mi_mapped_codes_in_map": sum(1 for m in SNOMED.values() if m == "MI"),
    "mi_mapped_codes": [c for c, m in SNOMED.items() if m == "MI"],
    "sttc_mapped_codes": sttc_codes,
    "hyp_mapped_codes": [c for c, m in SNOMED.items() if m == "HYP"],
    "norm_mapped_codes": [c for c, m in SNOMED.items() if m == "NORM"],
    "cd_mapped_codes": [c for c, m in SNOMED.items() if m == "CD"],
}
(OUT / "W1K_full_code_histogram.json").write_text(
    json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")

print("")
for p in sorted(OUT.glob("W1K_*")):
    print("wrote", p)
