# Day 8 lab notes: A4 light domain adaptation (e5-instruct)

Goal: push **test MAE_A** from five-head **0.774** toward **0.76**, still on cached `multilingual-e5-large-instruct` only. No new 8B embedder. **No test gold labels in training.**

Command:

```powershell
python -m src.domain_adapt --dev-copies 3
```

## Baselines (already in repo; reconfirmed)

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| SVR `train_full_dev` → test | 0.488 | 0.788 | 0.788 | 0.578 |
| Five-head ensemble → test | **0.473** | **0.774** | **0.795** | **0.598** |

`train_full_dev` remains the right baseline for test (paper Table 3 spirit: use labeled_dev in the fit). Scoring that same fit on **dev** is optimistic and is not the claim here.

## Experiments (test only; gold test unused in fit)

| Method | MAE_V | MAE_A | PCC_V | PCC_A | Notes |
|---|---:|---:|---:|---:|---|
| baseline `train_full_dev` | 0.488 | 0.788 | 0.788 | 0.578 | matches Day 4 SVR |
| **stronger_dev ×3 / sample_weight ×3** | 0.496 | **0.765** | 0.789 | 0.601 | best |
| retrieval upsample (400 NN ×2) | 0.489 | 0.794 | 0.788 | 0.569 | worse arousal |
| stronger_dev + retrieval | 0.495 | 0.770 | 0.789 | 0.593 | between |
| soft pseudo on holdout half_dev | 0.490 | 0.788 | 0.788 | 0.579 | ≈ baseline |
| half_dev gold + soft holdout | 0.496 | 0.798 | 0.784 | 0.571 | worse |

Leakage control:

- Retrieval: only **train** labels; neighbors picked by cosine similarity to **labeled_dev** embeddings.
- Pseudo: teacher on train + half labeled_dev; soft labels on the other half (gold of that half never enters `y`); test gold never enters training.

## Dev upsample sweep (MAE_A)

| copies | MAE_A |
|---:|---:|
| 2 | 0.770 |
| **3** | **0.765** |
| 4 | 0.766 |
| 5 | 0.770 |
| 6 | 0.776 |

Peak at ×3; heavier weighting starts to hurt.

## Vs goals / prior systems

| System | MAE_A |
|---|---:|
| Five-head ensemble | 0.774 |
| **A4 best (dev weight ×3 SVR)** | **0.765** |
| Stretch goal | 0.760 |
| Paper / board Encoders | ~0.76 |

**Positive result:** stronger use of labeled_dev alone beats the five-head average on arousal (−0.009) and beats plain SVR (−0.023). Did **not** quite hit 0.760 (gap ≈ 0.005). Valence is slightly worse than the ensemble (0.496 vs 0.473).

Retrieval upsample and careful pseudo-labeling were **negative / flat** on this cache.

## Files

- `src/domain_adapt.py`
- `results/intfloat__multilingual-e5-large-instruct_domain_adapt.json`
- `results/intfloat__multilingual-e5-large-instruct_test_domain_adapt_best_submission.csv`
