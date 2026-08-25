# Day 3 lab notes

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/).  
Setup: `multilingual-e5-large-instruct` + SVR (C=10, ε=0.2) on RTX 3070 8GB.  
At this point I was **not** planning DeepSeek/Llama encoders, five-encoder fusion, or CustomResNet yet.

Numbers rounded to 3 decimals. Raw JSON is in `results/`.

## Data (fixed on Day 1)

| Split | Source | n | Note |
|---|---|---:|---|
| train | CVAT 5 folds merged | 2954 | skip `CVAT_all_SD.csv` (broken rows) |
| dev | DSAMST validation (labeled) | 994 | matches paper |
| test | DSAMST test | 1541 | scored on Day 4; see `day4.md` |

Embeddings use an `Instruct:` prefix and `max_length=512`. Dim = 1024. No L2 by default.

## Table 1 style: train mixes

Paper Table 1 uses **DeepSeek**, not e5. I used the same train mixes and SVR, so trends matter more than exact matches.

| Mix | Encoder | MAE_V | MAE_A | PCC_V | PCC_A | n_train / n_eval |
|---|---|---:|---:|---:|---:|---|
| train only | paper DeepSeek | 0.528 | 0.914 | 0.711 | 0.483 | — |
| train only | our e5 | 0.586 | 1.120 | 0.725 | 0.490 | 2954 / 994 |
| half_dev | paper DeepSeek | 0.524 | 0.809 | 0.754 | 0.538 | — |
| half_dev | our e5 | **0.541** | **1.044** | **0.756** | **0.526** | 3451 / 497 |
| full_dev | paper DeepSeek | 0.524 | 0.809 | 0.754 | 0.538 | — |
| full_dev | our e5 | 0.497 | 0.984 | 0.788 | 0.590 | 3948 / 994 (dev already in train) |

Same pattern as the paper: half_dev beats train-only. Valence is close; arousal is higher by ~0.2.

## Table 2 style: regressors (half_dev)

Paper Table 2 = DeepSeek + half_dev. Ours = e5 + half_dev.

| Model | Source | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| SVR | paper DeepSeek | 0.524 | 0.809 | 0.754 | 0.538 |
| LGBM | paper DeepSeek | 0.519 | 0.852 | 0.740 | 0.472 |
| XGBoost | paper DeepSeek | 0.526 | 0.857 | 0.724 | 0.457 |
| SVR | our e5 | 0.541 | **1.044** | 0.756 | 0.526 |
| LGBM | our e5 | 0.540 | 1.051 | 0.752 | 0.496 |
| XGBoost | our e5 | 0.537 | 1.046 | 0.751 | 0.506 |

Trees did not beat SVR here. On `train_full_dev`, LGBM/XGB look almost perfect (MAE≈0) because they memorize the same labeled_dev — not useful.

## Table 3 style: encoders (paper: SVR + full_dev)

| Encoder | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| paper DeepSeek-R1 | 0.524 | 0.809 | 0.754 | 0.538 |
| paper e5-instruct | **0.523** | **0.807** | 0.742 | 0.539 |
| paper e5 (no instruct) | 0.554 | 0.810 | 0.726 | 0.531 |
| our e5-instruct (full_dev, optimistic) | 0.497 | **0.984** | 0.788 | 0.590 |

Paper’s e5 arousal 0.807 is **full_dev**. Their half_dev 0.809 is **DeepSeek**. So I should not compare our half-dev 1.044 directly to 0.807. Even with full_dev, our MAE_A is still 0.984.

## Small checks (half_dev, e5)

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| SVR paper defaults | 0.541 | 1.044 | 0.756 | 0.526 |
| SVR best grid (separate V/A) | 0.540 | 1.033 | 0.755 | 0.528 |
| LGBM | 0.540 | 1.051 | 0.752 | 0.496 |
| XGBoost | 0.537 | 1.046 | 0.751 | 0.506 |
| SVR + L2 | 0.541 | 1.044 | 0.756 | 0.526 |

L2 barely changes anything. I did not try an arousal-only instruct prompt (kept scope small).

## What I think

- Valence is already usable (0.541 vs paper e5 full_dev 0.523).
- Arousal does not move much when I tune SVR or switch trees.
- The hard part seems to be the embedding / domain gap (general CVAT → medical text), not SVR knobs.
- Chasing DeepSeek on 8GB this week did not look worth it yet.

Working baseline: **e5-instruct + SVR + half_dev**  
0.541 / 1.044 / 0.756 / 0.526

## Later

Day 4–5 are done; see `day4.md` and `report.md`.
