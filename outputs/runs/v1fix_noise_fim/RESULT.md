# M4 — Fisher-Information Noise-Model Sensitivity

## Per-lead noise (scaled signal, isoelectric baseline first 20 samples)

| lead | noise std |
|---|---|
| I | 0.53723 |
| II | 0.57785 |
| III | 0.52646 |
| aVR | 0.55394 |
| aVL | 0.48105 |
| aVF | 0.58917 |
| V1 | 0.54774 |
| V2 | 0.28412 |
| V3 | 0.28580 |
| V4 | 0.29430 |
| V5 | 0.32607 |
| V6 | 0.38936 |

Noise varies ~2.1× across leads (max/min).

## D-optimal greedy selection: top-5 leads

| rank | W = I (identity) | W = Σ⁻¹ (empirical noise) |
|---|---|---|
| 1 | V1 | V2 |
| 2 | V4 | V5 |
| 3 | I | V3 |
| 4 | V2 | V4 |
| 5 | V5 | V1 |


## 3-lead set comparison
- W=I top-3 = ['V1', 'V4', 'I']
- W=Σ⁻¹ top-3 = ['V2', 'V5', 'V3']
- Same unordered top-3 = False

## Conclusion
The empirical noise model changed the unordered identity-noise top-3 set.
Exact montage claims are conditional on this decoder, fold-9 design subset, parameter scaling, gauge choice, and noise model.

## Method
- Empirical noise variance: per lead, the variance of the first 20 samples of the scaled signal (100 Hz, TP/PR isoelectric segment)
- Jacobian: as in the manuscript, computed with jacfwd on the locked decoder
- FIM subset: 400 class-balanced fold-9 records; independent lead set: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
- Gauge: global_gain held fixed in the local analysis; FIM dimension = 46
- Regularization: lambda = 0.001
- Script: scripts/53_noise_model_sensitivity.py
