"""v3 Phase 1 robustness wrap-up — per-class optimal leads + 2nd-place margin + complementarity.

Frozen boundary-B; read-only; 8 independent leads {I,II,V1..V6}; fixed 400-sample subset (100/class).
A) Per-class FIM (NORM/MI/STTC/CD): exhaustive D/A/E-optimal 3-set; is it I+V1+V4? vs clinical gap.
B) Global D-optimal k=3 top-3 sets + (1st-2nd) margin vs bootstrap gap width.
C) Marginal contribution of each lead within the optimal set (low redundancy / spatial complementarity).

Run: "$PY" PUBLIC/scripts/52_leadset8_byclass.py
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
try:
    import torch   # noqa
except Exception:
    pass
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

LAM = 1e-3
IND = [CANONICAL_LEADS.index(x) for x in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
INAME = {l: CANONICAL_LEADS[l] for l in IND}
CLIN = tuple(CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5"))
OPT = tuple(CANONICAL_LEADS.index(x) for x in ("I", "V1", "V4"))


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed); idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)
    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0)); return y12[0].reshape(-1)
    return torch.stack([jacfwd(f)(thetas[s].to(device)) * r for s in range(thetas.shape[0])]).detach()


def logdet(F): return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])
def trinv(F):  return float(np.trace(np.linalg.inv(F + LAM * np.eye(F.shape[0]))))
def lmin(F):   return float(np.linalg.eigvalsh(F + LAM * np.eye(F.shape[0]))[0])
def setname(S): return "+".join(INAME[i] for i in S)


def main():
    ap = argparse.ArgumentParser()
    # The locked lineage, not a path: this default used to name the superseded
    # eikonal_B weights and was corrected only by a driver's monkey patch.
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--npz", default="data/processed/ptbxl_medianbeat_clean_100hz.npz")
    # None -> the published run for the locked seed; the old default was the
    # pre-v1fix phase1 run, which the paper does not report.
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / (a.out or repro.outdir("runs/v1fix_byclass_s42")); (out / "tables").mkdir(parents=True, exist_ok=True)

    data = load_processed(ROOT / a.npz)
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(ROOT / (a.ckpt or repro.lineage.checkpoint(42)), n_t, 3, dev)
    idx = fixed_subset(label, fold); lab_sub = label[idx]
    theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    thetas = torch.from_numpy(theta).to(dev)
    J_all = compute_J_all(dec, thetas, dec.pspace.radius, dev)
    G = {l: torch.einsum("nrp,nrq->npq", J_all[:, torch.tensor(lead_rows([l], n_t), device=dev), :],
                         J_all[:, torch.tensor(lead_rows([l], n_t), device=dev), :]).cpu().numpy() for l in IND}
    combos = list(itertools.combinations(IND, 3))
    print(f"[byclass] ckpt={(a.ckpt or repro.lineage.checkpoint(42))} n={len(idx)} combos={len(combos)}")

    def F_of(S, rows=None):
        sl = slice(None) if rows is None else rows
        return sum(G[l][sl].mean(0) for l in S)

    # ---------- A. per-class optimal ----------
    rowsA = []
    for c, cn in enumerate(TARGET_CLASSES):
        m = np.where(lab_sub == c)[0]
        D = {S: logdet(F_of(S, m)) for S in combos}
        A = {S: trinv(F_of(S, m)) for S in combos}
        E = {S: lmin(F_of(S, m)) for S in combos}
        bD = max(D, key=D.get); bA = min(A, key=A.get); bE = max(E, key=E.get)
        gap = D[bD] - logdet(F_of(CLIN, m))   # best-D vs clinical (same class FIM)
        rowsA.append({"class": cn, "n": len(m),
                      "D_opt": setname(bD), "A_opt": setname(bA), "E_opt": setname(bE),
                      "DAE_agree": bD == bA == bE, "D_eq_I+V1+V4": bD == OPT,
                      "best_D_logdet": D[bD], "clinical_logdet": logdet(F_of(CLIN, m)),
                      "best_minus_clinical": gap})
        print(f"[A] {cn:5s} D*={setname(bD):10s} A*={setname(bA):10s} E*={setname(bE):10s} "
              f"=I+V1+V4:{bD==OPT}  best-clin gap={gap:.2f}")
    pd.DataFrame(rowsA).to_csv(out / "tables" / "byclass_optimal_DAE.csv", index=False)

    # ---------- B. global top-3 + margin ----------
    Dall = sorted(((logdet(F_of(S)), S) for S in combos), reverse=True)
    top = Dall[:3]
    rowsB = [{"rank": i + 1, "set": setname(S), "logdet": d,
              "margin_from_1st": d - top[0][0]} for i, (d, S) in enumerate(top)]
    margin12 = top[0][0] - top[1][0]
    pd.DataFrame(rowsB).to_csv(out / "tables" / "global_top3_Doptimal.csv", index=False)
    print(f"[B] top3: " + " | ".join(f"#{r['rank']} {r['set']}={r['logdet']:.2f}" for r in rowsB))
    print(f"[B] margin (1st-2nd) = {margin12:.3f}  (bootstrap gap width ~0.35 from v05 CI[10.64,10.99])")

    # ---------- C. marginal contribution within optimal set (redundancy/complementarity) ----------
    full = logdet(F_of(OPT))
    rowsC = []
    for l in OPT:
        rest = tuple(x for x in OPT if x != l)
        rowsC.append({"lead_dropped": INAME[l], "logdet_pair": logdet(F_of(rest)),
                      "marginal_gain_of_lead": full - logdet(F_of(rest))})
    pd.DataFrame(rowsC).to_csv(out / "tables" / "optimal_set_marginal_contrib.csv", index=False)
    print("[C] marginal logdet gain of each lead in I+V1+V4: " +
          ", ".join(f"{r['lead_dropped']}=+{r['marginal_gain_of_lead']:.2f}" for r in rowsC))
    print(f"[byclass] DONE -> {out}")


if __name__ == "__main__":
    main()
