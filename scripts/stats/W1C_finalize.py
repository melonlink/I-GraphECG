"""W1C step 3: derived summary columns, the T-window censoring check, and the headline JSON.

Reads   runs/v7rev_stats/W1C_descriptor_fidelity.csv, W1C_summary.csv, W1C_domain_rollup.csv
        runs/v1fix_r3_s{seed}/tables/round3_features.csv          (for the Tptime censoring check)
Writes  runs/v7rev_stats/W1C_summary.csv     + pearson_minus_spearman, recon_over_obs_mean
        runs/v7rev_stats/W1C_Tptime_window_saturation.csv
        runs/v7rev_stats/W1C_headline.json

Provenance note. The frozen versions of these three files were produced on 2026-09-03 by a
step that was not saved as a script. This file reconstructs it from the frozen outputs and
is verified against them byte for byte (numbers) and by JSON equality (headline). The
explanatory sentences in the headline are reproduced as templates over the same numbers.
The summary is parsed with float_precision="round_trip", which is what makes the two
derived columns reproduce textually.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro

RUNS = repro.runs()
OUT = repro.outdir("runs/v7rev_stats")
SEEDS = [42, 1, 2, 3, 4]

per = pd.read_csv(OUT / "W1C_descriptor_fidelity.csv", float_precision="round_trip")
s = pd.read_csv(OUT / "W1C_summary.csv", float_precision="round_trip")
dom = pd.read_csv(OUT / "W1C_domain_rollup.csv", float_precision="round_trip")

# ---- derived summary columns -------------------------------------------------------
s = s.drop(columns=[c for c in ("pearson_minus_spearman", "recon_over_obs_mean") if c in s.columns])
s["pearson_minus_spearman"] = s["pearson_r_mean"] - s["spearman_rho_mean"]
s["recon_over_obs_mean"] = s["recon_mean_mean"] / s["obs_mean_mean"]
s.to_csv(OUT / "W1C_summary.csv", index=False, encoding="utf-8")

# ---- T-window censoring: Tptime is defined on a discrete grid ending at 440 ms ---------
NOTE = "Tptime is censored to the T window [160,440] ms (T_WIN=(160,450) ms at fs=100 Hz)"
rows = []
for seed in SEEDS:
    df = pd.read_csv(RUNS / f"v1fix_r3_s{seed}" / "tables" / "round3_features.csv",
                     float_precision="round_trip")
    te = df[df["fold"].astype(int) == 10]
    rows.append({
        "seed": seed, "n_fold10": int(len(te)),
        "frac_recon_Tptime_maxabs_at_440ms": float((te["Tptime_maxabs"] == 440).mean()),
        "frac_obs_Tptime_maxabs_at_440ms": float((te["obs_Tptime_maxabs"] == 440).mean()),
        "frac_recon_Tptime_mean_ge_400ms": float((te["Tptime_mean"] >= 400).mean()),
        "frac_obs_Tptime_mean_ge_400ms": float((te["obs_Tptime_mean"] >= 400).mean()),
        "recon_Tptime_maxabs_n_unique": int(te["Tptime_maxabs"].nunique()),
        "obs_Tptime_maxabs_n_unique": int(te["obs_Tptime_maxabs"].nunique()),
        "note": NOTE,
    })
sat = pd.DataFrame(rows)
sat.to_csv(OUT / "W1C_Tptime_window_saturation.csv", index=False, encoding="utf-8")

# ---- headline ---------------------------------------------------------------------
r = s["pearson_r_mean"]
hi, lo = s[r >= 0.90], s[r < 0.50]
mid = s[(r >= 0.50) & (r < 0.80)]
tsd = s[s["pair"] == "T_sign_discordance_rate"].iloc[0]
AMP = ("Tpeak", "Tarea", "Tabsarea", "STmean", "STslope", "Tlowamp")


def band_block(name):
    b = s[s["fidelity_band"] == name]
    return {"n": int(len(b)), "pairs": sorted(b["pair"].tolist()),
            "r_range": [float(b["pearson_r_mean"].min()), float(b["pearson_r_mean"].max())]}


# the sorted mean-r ladder and its two largest gaps
rr = r.to_numpy()
gap = rr[:-1] - rr[1:]
order = np.argsort(-gap)
g1, g2 = int(order[0]), int(order[1])
definition = (f"largest gap in the sorted five-seed mean-r ladder is {rr[g1]:.3f}->{rr[g1 + 1]:.3f} "
              f"({gap[g1]:.4f}) between {s['pair'].iloc[g1]} and {s['pair'].iloc[g1 + 1]}; "
              f"second largest {rr[g2]:.3f}->{rr[g2 + 1]:.3f} ({gap[g2]:.4f}). "
              f"Bands: A r>=0.80, B 0.45<=r<0.80, C r<0.45.")

A, B, C = band_block("A_faithful"), band_block("B_partial"), band_block("C_not_recovered")
fam_span = (s.groupby("family")["pearson_r_mean"].agg(lambda x: x.max() - x.min())
            .sort_values(ascending=False))
volt = s[s["domain"] == "voltage"]
nonvolt = s[s["domain"].isin(["timing", "shape", "polarity"])]
tl = s.set_index("pair")["pearson_r_mean"]
volt_fam_span = fam_span[[f for f in fam_span.index if f in ("Tpeak", "Tarea", "Tabsarea", "STmean", "STslope")]]
caveat = (f"Not a clean amplitude-vs-timing split. All {len(volt)} voltage-scale pairs "
          f"(Tpeak/Tarea/Tabsarea/STmean/STslope) sit at r>={volt['pearson_r_mean'].min():.3f}, and all "
          f"{len(nonvolt)} timing/shape/polarity pairs at r<={nonvolt['pearson_r_mean'].max():.3f}, but the "
          f"{int((s['family'] == 'Tlowamp').sum())}-member Tlowamp family - a monotone saturating transform of "
          f"|T peak| amplitude, exp(-|Tpeak|/0.2) - straddles the divide: Tlowamp_mean {tl['Tlowamp_mean']:.3f} "
          f"(band A) while Tlowamp_maxabs {tl['Tlowamp_maxabs']:.3f} (band B, only "
          f"{tl['Tlowamp_maxabs'] - tl['T_sign_discordance_rate']:.2f} above the polarity descriptor). "
          f"Tlowamp spans {fam_span['Tlowamp']:.2f} in r across its own four cross-lead aggregations, wider "
          f"than any voltage family (widest voltage family span is {volt_fam_span.index[0]}, "
          f"{volt_fam_span.iloc[0]:.3f}).")

# The censoring numbers are taken from the CSV as written and re-read with pandas' default
# parser, not from the in-memory frame: that is how the frozen headline was computed (its
# obs fraction sits one ulp from 116/1594, the default parser's reading of the CSV text).
sat_disk = pd.read_csv(OUT / "W1C_Tptime_window_saturation.csv")
pin_mean = float(sat_disk["frac_recon_Tptime_maxabs_at_440ms"].mean())
pin_sd = float(sat_disk["frac_recon_Tptime_maxabs_at_440ms"].std(ddof=1))
pin_obs = float(sat_disk["frac_obs_Tptime_maxabs_at_440ms"].mean())   # identical across seeds
censoring = {
    "frac_recon_Tptime_maxabs_pinned_at_440ms_mean": pin_mean,
    "frac_recon_Tptime_maxabs_pinned_at_440ms_sd": pin_sd,
    "frac_obs_Tptime_maxabs_pinned_at_440ms": pin_obs,
    "note": (f"T_WIN=(160,450) ms at fs=100 Hz gives a discrete grid 160..440 ms. The reconstruction "
             f"pins Tptime_maxabs at the 440 ms edge in {100 * pin_mean:.1f}+-{100 * pin_sd:.1f}% of fold-10 "
             f"records vs {100 * pin_obs:.1f}% for the observation, so the near-zero r={tl['Tptime_maxabs']:.3f} "
             f"for Tptime_maxabs is partly a ceiling/censoring artifact of the descriptor definition, not "
             f"purely a reconstruction failure."),
}
DOMAIN_KEY = {"voltage": "voltage", "voltage-derived (saturating)": "voltage_derived_Tlowamp",
              "polarity": "polarity", "shape": "shape", "timing": "timing"}
rollup = {f"{DOMAIN_KEY[row.domain]}_n{int(row.n_pairs)}": round(float(row.r_mean), 4)
          for row in dom.sort_values("r_mean", ascending=False).itertuples()}

head = {
    "task": "W1C", "n_pairs": int(len(s)), "n_test_fold10": int(per["n"].iloc[0]), "seeds": SEEDS,
    "n_r_ge_090": int(len(hi)),
    "n_r_080_090": int(((r >= 0.80) & (r < 0.90)).sum()),
    "n_r_050_080": int(((r >= 0.50) & (r < 0.80)).sum()),
    "n_r_lt_050": int(len(lo)),
    "best": {"pair": s["pair"].iloc[0], "r_mean": float(r.iloc[0]), "r_sd": float(s["pearson_r_sd"].iloc[0])},
    "worst": {"pair": s["pair"].iloc[-1], "r_mean": float(r.iloc[-1]), "r_sd": float(s["pearson_r_sd"].iloc[-1])},
    "control_T_sign_discordance_rate": {
        "r_five_seed_mean": float(tsd["pearson_r_mean"]), "r_five_seed_sd": float(tsd["pearson_r_sd"]),
        "r_seed42": float(per.loc[(per["pair"] == "T_sign_discordance_rate") & (per["seed"] == 42), "pearson_r"].iloc[0])},
    "middle_band_pairs": sorted(mid["pair"].tolist()),
    "amplitude_pairs_in_low_band": sorted([x for x in lo["pair"] if x.startswith(AMP)]),
    "bands_empirical": {"definition": definition, "A_faithful": A, "B_partial": B, "C_not_recovered": C},
    "dichotomy_caveat": caveat,
    "Tptime_window_censoring": censoring,
    "domain_rollup_mean_r": rollup,
    "within_family_r_span": {k: round(float(v), 4) for k, v in fam_span.items()},
}
(OUT / "W1C_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
print(json.dumps(head, indent=2)[:1500])
print("\nwrote W1C_summary.csv (+2 columns), W1C_Tptime_window_saturation.csv, W1C_headline.json")
