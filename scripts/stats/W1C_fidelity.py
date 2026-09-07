"""W1C step 1: reconstruction-domain vs observation-domain descriptor fidelity.

For every reconstructed-ECG descriptor that has an obs_-prefixed twin, compute on the
fold-10 test set: Pearson r, Spearman rho, Bland-Altman bias and 95% limits of agreement.
Repeated for the five published seeds (v1fix_r3_s{42,1,2,3,4}); the headline is the
five-seed mean +- SD (ddof=1).

Reads   runs/v1fix_r3_s{seed}/tables/round3_features.csv and feature_provenance.csv
Writes  runs/v7rev_stats/W1C_descriptor_fidelity.csv   per seed x pair
        runs/v7rev_stats/W1C_summary.csv               per pair, five-seed summary
        runs/v7rev_stats/W1C_controls.json
Then run W1C_bands.py and W1C_finalize.py, which extend W1C_summary.csv in place.

This generator was written for the reviewer response and ran from a scratch directory
with absolute paths; it is recorded here, resolving through igraphecg.repro, so that the
Supplementary descriptor-fidelity table has its producer in the archive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro

RUNS = repro.runs()
OUT = repro.outdir("runs/v7rev_stats")
SEEDS = [42, 1, 2, 3, 4]

# descriptor "family" -> unit, for reporting
UNITS = {
    "Tpeak": "mV", "Tptime": "ms", "Tarea": "mV*s", "Tabsarea": "mV*s",
    "Twidth": "fraction of T window", "Tasym": "dimensionless (-1..1)",
    "Tlowamp": "dimensionless (0..1)", "STmean": "mV", "STslope": "mV/sample",
    "T_sign_discordance_rate": "fraction of leads",
}


# which aggregation each descriptor is (mean/std/maxabs/range across the 12 leads)
def family_of(name: str) -> str:
    if name == "T_sign_discordance_rate":
        return "T_sign_discordance_rate"
    return name.rsplit("_", 1)[0]


def kind_of(name: str) -> str:
    if name == "T_sign_discordance_rate":
        return "rate"
    return name.rsplit("_", 1)[1]


def load(seed: int) -> pd.DataFrame:
    return pd.read_csv(RUNS / f"v1fix_r3_s{seed}" / "tables" / "round3_features.csv")


def pairs_from_provenance(seed: int = 42) -> list[str]:
    prov = pd.read_csv(RUNS / f"v1fix_r3_s{seed}" / "tables" / "feature_provenance.csv")
    recon = [f for f in prov.loc[prov["source"] == "reconstructed_ecg", "feature"]]
    obs = set(prov.loc[prov["source"] == "observed_ecg", "feature"])
    out = [f for f in recon if f"obs_{f}" in obs]
    unmatched_recon = [f for f in recon if f"obs_{f}" not in obs]
    unmatched_obs = [f for f in obs if f[4:] not in set(recon)]
    print(f"[pairs] recon={len(recon)} obs={len(obs)} matched={len(out)} "
          f"unmatched_recon={unmatched_recon} unmatched_obs={unmatched_obs}")
    return out


def stats_for_pair(rec: np.ndarray, obs: np.ndarray) -> dict:
    m = np.isfinite(rec) & np.isfinite(obs)
    r_, o_ = rec[m], obs[m]
    n = int(m.sum())
    d = r_ - o_
    mean_xy = 0.5 * (r_ + o_)
    bias = float(d.mean())
    sd_d = float(d.std(ddof=1))
    res = {
        "n": n,
        "pearson_r": float("nan"), "pearson_p": float("nan"),
        "spearman_rho": float("nan"), "spearman_p": float("nan"),
        "recon_mean": float(r_.mean()), "recon_sd": float(r_.std(ddof=1)),
        "obs_mean": float(o_.mean()), "obs_sd": float(o_.std(ddof=1)),
        "ba_bias": bias, "ba_sd_diff": sd_d,
        "ba_loa_lower": bias - 1.96 * sd_d, "ba_loa_upper": bias + 1.96 * sd_d,
        "ba_loa_width": 2 * 1.96 * sd_d,
        "ba_bias_over_obs_sd": float("nan"),
        "ba_loa_width_over_obs_sd": float("nan"),
        "ba_prop_bias_slope": float("nan"),  # regression of diff on mean (proportional bias)
    }
    o_sd = res["obs_sd"]
    if np.isfinite(o_sd) and o_sd > 0:
        res["ba_bias_over_obs_sd"] = bias / o_sd
        res["ba_loa_width_over_obs_sd"] = res["ba_loa_width"] / o_sd
    if n >= 3 and r_.std() > 0 and o_.std() > 0:
        pr = stats.pearsonr(r_, o_)
        sr = stats.spearmanr(r_, o_)
        res["pearson_r"], res["pearson_p"] = float(pr[0]), float(pr[1])
        res["spearman_rho"], res["spearman_p"] = float(sr[0]), float(sr[1])
    if n >= 3 and mean_xy.std() > 0:
        lr = stats.linregress(mean_xy, d)
        res["ba_prop_bias_slope"] = float(lr.slope)
    return res


def main() -> int:
    pair_names = pairs_from_provenance(42)
    print(f"[pairs] n_pairs = {len(pair_names)}")

    rows = []
    controls = {}
    for seed in SEEDS:
        df = load(seed)
        assert len(df) == 15709, f"seed {seed}: expected 15709 rows, got {len(df)}"
        te = df["fold"].astype(int) == 10
        sub = df.loc[te]
        controls[f"n_fold10_s{seed}"] = int(len(sub))
        print(f"[seed {seed}] N_total={len(df)} N_fold10={len(sub)}")
        for name in pair_names:
            rec = sub[name].to_numpy(dtype=float)
            obs = sub[f"obs_{name}"].to_numpy(dtype=float)
            st = stats_for_pair(rec, obs)
            rows.append({"seed": seed, "pair": name, "recon_feature": name,
                         "obs_feature": f"obs_{name}", "family": family_of(name),
                         "aggregation": kind_of(name),
                         "unit": UNITS.get(family_of(name), "unknown"), **st})

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "W1C_descriptor_fidelity.csv", index=False, encoding="utf-8")

    agg_cols = ["pearson_r", "spearman_rho", "ba_bias", "ba_sd_diff", "ba_loa_lower",
                "ba_loa_upper", "ba_loa_width", "ba_bias_over_obs_sd",
                "ba_loa_width_over_obs_sd", "ba_prop_bias_slope",
                "recon_mean", "recon_sd", "obs_mean", "obs_sd"]
    g = per.groupby("pair", sort=False)
    summ = pd.DataFrame({"pair": list(g.groups.keys())})
    meta = per.drop_duplicates("pair").set_index("pair")
    summ["family"] = summ["pair"].map(meta["family"])
    summ["aggregation"] = summ["pair"].map(meta["aggregation"])
    summ["unit"] = summ["pair"].map(meta["unit"])
    summ["n_fold10"] = summ["pair"].map(g["n"].mean())
    for c in agg_cols:
        summ[f"{c}_mean"] = summ["pair"].map(g[c].mean())
        summ[f"{c}_sd"] = summ["pair"].map(g[c].std(ddof=1))
    # per-seed pearson spread, useful for the seed-42 caveat
    pv = per.pivot(index="pair", columns="seed", values="pearson_r")
    for s in SEEDS:
        summ[f"pearson_r_s{s}"] = summ["pair"].map(pv[s])
    summ["pearson_r_min"] = summ["pair"].map(pv.min(axis=1))
    summ["pearson_r_max"] = summ["pair"].map(pv.max(axis=1))
    summ = summ.sort_values("pearson_r_mean", ascending=False).reset_index(drop=True)
    summ.insert(0, "rank_by_mean_pearson_r", np.arange(1, len(summ) + 1))
    summ.to_csv(OUT / "W1C_summary.csv", index=False, encoding="utf-8")

    # ---- controls ----
    tsd = summ.loc[summ["pair"] == "T_sign_discordance_rate"].iloc[0]
    controls["T_sign_discordance_rate_pearson_r_s42"] = float(tsd["pearson_r_s42"])
    controls["T_sign_discordance_rate_pearson_r_5seed_mean"] = float(tsd["pearson_r_mean"])
    controls["T_sign_discordance_rate_pearson_r_5seed_sd"] = float(tsd["pearson_r_sd"])
    controls["n_pairs"] = len(pair_names)
    with open(OUT / "W1C_controls.json", "w", encoding="utf-8") as fh:
        json.dump(controls, fh, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print("\n=== summary (sorted by five-seed mean Pearson r) ===")
    print(summ[["rank_by_mean_pearson_r", "pair", "unit", "pearson_r_mean", "pearson_r_sd",
                "spearman_rho_mean", "spearman_rho_sd", "ba_bias_mean", "ba_bias_sd",
                "ba_loa_lower_mean", "ba_loa_upper_mean",
                "ba_bias_over_obs_sd_mean", "ba_loa_width_over_obs_sd_mean"]].to_string(index=False))
    print("\n=== controls ===")
    print(json.dumps(controls, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
