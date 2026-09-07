"""W1C step 2: fidelity bands, domain roll-up, seed-42 vs five-seed rank stability.

Reads   runs/v7rev_stats/W1C_descriptor_fidelity.csv, W1C_summary.csv   (from W1C_fidelity.py)
Writes  runs/v7rev_stats/W1C_domain_rollup.csv
        runs/v7rev_stats/W1C_summary.csv   extended in place with `domain` and `fidelity_band`
Then run W1C_finalize.py.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro

OUT = repro.outdir("runs/v7rev_stats")
SEEDS = [42, 1, 2, 3, 4]

per = pd.read_csv(OUT / "W1C_descriptor_fidelity.csv")
summ = pd.read_csv(OUT / "W1C_summary.csv")

DOMAIN = {"Tpeak": "voltage", "Tarea": "voltage", "Tabsarea": "voltage",
          "STmean": "voltage", "STslope": "voltage",
          "Tptime": "timing", "Twidth": "shape", "Tasym": "shape",
          "Tlowamp": "voltage-derived (saturating)",
          "T_sign_discordance_rate": "polarity"}
summ["domain"] = summ["family"].map(DOMAIN)

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 120)

print("=== per-seed Pearson r, ordered by five-seed mean ===")
cols = ["rank_by_mean_pearson_r", "pair", "domain", "pearson_r_mean", "pearson_r_sd",
        "pearson_r_s42", "pearson_r_s1", "pearson_r_s2", "pearson_r_s3", "pearson_r_s4",
        "pearson_r_min", "pearson_r_max"]
print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== gaps in the sorted mean-r ladder ===")
r = summ["pearson_r_mean"].to_numpy()
gaps = pd.DataFrame({
    "above_rank": summ["rank_by_mean_pearson_r"].to_numpy()[:-1],
    "above_pair": summ["pair"].to_numpy()[:-1],
    "below_pair": summ["pair"].to_numpy()[1:],
    "r_above": r[:-1], "r_below": r[1:], "gap": r[:-1] - r[1:]})
print(gaps.sort_values("gap", ascending=False).head(8).to_string(index=False,
      float_format=lambda v: f"{v:.4f}"))

print("\n=== domain-level aggregation (five-seed mean r per pair, then over pairs) ===")
dom = summ.groupby("domain").agg(
    n_pairs=("pair", "size"),
    r_min=("pearson_r_mean", "min"), r_median=("pearson_r_mean", "median"),
    r_max=("pearson_r_mean", "max"), r_mean=("pearson_r_mean", "mean")).sort_values("r_mean", ascending=False)
print(dom.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n=== family-level ===")
fam = summ.groupby("family").agg(
    domain=("domain", "first"), n=("pair", "size"),
    r_min=("pearson_r_mean", "min"), r_max=("pearson_r_mean", "max"),
    r_mean=("pearson_r_mean", "mean"),
    loa_w_over_sd=("ba_loa_width_over_obs_sd_mean", "mean")).sort_values("r_mean", ascending=False)
print(fam.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n=== seed-42-only ranking vs five-seed ranking (rank shifts) ===")
s42 = summ[["pair", "pearson_r_s42", "pearson_r_mean", "rank_by_mean_pearson_r"]].copy()
s42["rank_s42"] = s42["pearson_r_s42"].rank(ascending=False).astype(int)
s42["rank_shift"] = s42["rank_s42"] - s42["rank_by_mean_pearson_r"]
print(s42.reindex(s42["rank_shift"].abs().sort_values(ascending=False).index).head(10)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== Bland-Altman detail, five-seed mean +- sd, native units ===")
bacols = ["rank_by_mean_pearson_r", "pair", "unit", "obs_mean_mean", "obs_sd_mean",
          "recon_mean_mean", "ba_bias_mean", "ba_bias_sd", "ba_sd_diff_mean",
          "ba_loa_lower_mean", "ba_loa_upper_mean", "ba_prop_bias_slope_mean"]
print(summ[bacols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# banded classification written out as a supplementary column set
def band(x):
    if x >= 0.80:
        return "A_faithful"
    if x >= 0.45:
        return "B_partial"
    return "C_not_recovered"


summ["fidelity_band"] = summ["pearson_r_mean"].apply(band)
summ.to_csv(OUT / "W1C_summary.csv", index=False, encoding="utf-8")
print("\n=== band counts ===")
print(summ.groupby(["fidelity_band", "domain"]).size().to_string())

dom.reset_index().to_csv(OUT / "W1C_domain_rollup.csv", index=False, encoding="utf-8")
print("\nwrote W1C_domain_rollup.csv and refreshed W1C_summary.csv with fidelity_band/domain")
