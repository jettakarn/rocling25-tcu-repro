# Table 1–2 cell alignment (DeepSeek-R1 FP16)

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/).  
Our encoder: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` FP16 mean-pool (`quantized=false`), SVR C=10 ε=0.2 unless noted.  
Artifacts: `results/deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_*.json`.

Paper Tables 1–2 are **DeepSeek + eval on labeled_dev** (Table 1 = train mixes; Table 2 = regressors on the half_dev mix). They are **not** the shared-task test board.

## Table 1 — training mixes (SVR)

| Mix | Paper DeepSeek | Ours DeepSeek-R1 FP16 | Δ MAE_V | Δ MAE_A |
|---|---|---|---:|---:|
| train only → full_dev | 0.528 / 0.914 / 0.711 / 0.483 | **0.677 / 1.118 / 0.659 / 0.397** | +0.149 | +0.204 |
| train_half_dev → holdout half | 0.524 / 0.809 / 0.754 / 0.538 | **0.578 / 1.047 / 0.728 / 0.459** | +0.054 | +0.238 |
| train_full_dev → full_dev (optimistic) | 0.524 / 0.809 / 0.754 / 0.538 | **0.453 / 0.862 / 0.829 / 0.669** | −0.071 | +0.053 |

Notes:

- Same **trend** as the paper: adding labeled_dev helps vs train-only.
- Our half_dev arousal is still high (~1.05), similar to the old e5 half_dev issue; paper’s half_dev MAE_A is 0.809.
- Our `full_dev_on_dev` valence is **better** than paper; arousal remains a bit worse (+0.05).
- Exact match is not expected (pooling / prompt / seed / tokenizer details unpublished).

## Table 2 — regressors (DeepSeek, `train_half_dev`)

| Model | Paper DeepSeek | Ours DeepSeek-R1 FP16 |
|---|---|---|
| SVR | 0.524 / 0.809 / 0.754 / 0.538 | **0.578 / 1.047 / 0.728 / 0.459** |
| LightGBM | 0.519 / 0.852 / 0.740 / 0.472 | **0.623 / 1.063 / 0.675 / 0.446** |
| XGBoost | 0.526 / 0.857 / 0.724 / 0.457 | **0.619 / 1.081 / 0.675 / 0.420** |
| CatBoost (ours only; paper Table 2 may omit) | — | **0.627 / 1.054 / 0.667 / 0.459** |

Same ranking spirit as the paper: **SVR beats trees** on this split. Absolute MAE_A gap remains (~0.2–0.25).

## Related (already done; not Table 1–2 cells)

| Item | Ours | Paper / board |
|---|---|---|
| Table 3 e5-instruct `full_dev_on_dev` | 0.497 / 0.984 | 0.523 / 0.807 |
| Table 3 DeepSeek `full_dev_on_dev` | 0.453 / 0.862 | 0.524 / 0.809 |
| Table 4 Models (e5 regressor mean) test | 0.473 / 0.774 | 0.495 / 0.802 |
| Table 4 Encoders (5-encoder) test | **0.470 / 0.758** | 0.463 / 0.759 |

## Takeaway

Table 1–2 **protocol cells are filled** for DeepSeek-R1 FP16 (SVR mixes + SVR/LGBM/XGB). Numbers do **not** reproduce paper cells tightly on half_dev arousal; the shared-task **test** encoder ensemble still matches Table 4 / board much more closely.
