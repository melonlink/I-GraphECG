"""W1F provenance: is the residual descriptor mismatch float32 noise, or a pipeline difference?

Evidence: rerun the FROZEN encoder over the clean records alone (exactly the frozen batch
composition) and over the full multi-label pool, and compare both to the frozen theta table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))

from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder

CKPT = repro.lineage.checkpoint(42)
FROZEN_RUN = repro.runs() / "v1fix_r3_s42"
OUT = repro.outdir("runs/v7rev_stats")

clean = np.load(REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
ml = np.load(REPO / "data/processed/ptbxl_medianbeat_multilabel_100hz.npz", allow_pickle=True)
med, iqr = clean["scaler_median"], clean["scaler_iqr"]
sc = RobustLeadScaler.from_dict({"median": med, "iqr": iqr})

fz = pd.read_csv(FROZEN_RUN / "tables" / "round3_features.csv")
assert np.array_equal(fz["record_id"].to_numpy(), clean["record_id"])
theta_frozen = fz[[f"theta_{i}" for i in range(47)]].to_numpy()

scaled_clean = sc.transform(clean["signal_12lead"].astype(np.float32))
scaled_all = sc.transform(ml["signal_12lead"].astype(np.float32))
pos = pd.Series(np.arange(len(ml["record_id"])), index=ml["record_id"]).reindex(
    clean["record_id"]).to_numpy().astype(int)

rows = []
for device in ["cuda", "cpu"]:
    enc, dec = load_encoder_decoder(CKPT, n_t=100, rank=3, device=device,
                                    scaler={"median": med, "iqr": iqr})
    th_a, _ = infer_theta(enc, dec, scaled_clean, device=device)
    th_b, _ = infer_theta(enc, dec, scaled_all, device=device)
    for name, diff in [("clean_only_pass_vs_frozen", np.abs(th_a - theta_frozen)),
                       ("pooled_pass_vs_frozen", np.abs(th_b[pos] - theta_frozen)),
                       ("pooled_vs_clean_only", np.abs(th_b[pos] - th_a))]:
        rows.append({"device": device, "comparison": name,
                     "max_abs_diff": float(diff.max()),
                     "median_abs_diff": float(np.median(diff)),
                     "p999_abs_diff": float(np.percentile(diff, 99.9)),
                     "frac_elements_nonzero": float((diff > 0).mean()),
                     "theta_table_sd": float(theta_frozen.std())})
        print(rows[-1])

df = pd.DataFrame(rows)
stab = pd.read_csv(FROZEN_RUN / "tables" / "stability_proxy_values.csv")
scale_rows = [{"quantity": "m_proxy", "frozen_sd": float(stab["m_proxy"].std()),
               "frozen_iqr": float(stab["m_proxy"].quantile(.75) - stab["m_proxy"].quantile(.25))}]
for c in ["ST_projected_L2", "Sync_rep", "Tend_disp", "D_act", "theta_0", "theta_23"]:
    scale_rows.append({"quantity": c, "frozen_sd": float(fz[c].std()),
                       "frozen_iqr": float(fz[c].quantile(.75) - fz[c].quantile(.25))})
out = pd.concat([df, pd.DataFrame(scale_rows)], axis=0, ignore_index=True)
out.to_csv(OUT / "W1F_multilabel_numeric_provenance.csv", index=False)
print("\nwrote", OUT / "W1F_multilabel_numeric_provenance.csv")
