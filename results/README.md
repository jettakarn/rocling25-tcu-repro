# Results: which files to trust

Official scores live under `results/`. This folder mixes **paper-path** numbers, **scratch** diagnostics, and **non-paper** extras. Use the lists below.

## Canonical (paper path — cite these)

| File | What it is |
|---|---|
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev.json` | e5-instruct + SVR → **test** (main single-model claim) |
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev_submission.csv` | matching submission |
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev_ensemble.json` | Table 4 **Models** spirit (SVR+LGBM+XGB+CatBoost+ResNet mean) |
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev_ensemble_submission.csv` | matching submission |
| `intfloat__multilingual-e5-large_test_train_full_dev.json` | e5-large (no instruct) → test (Table 3 row) |
| `intfloat__multilingual-e5-large-instruct_train_full_dev.json` | SVR `full_dev_on_dev` (Table 3 spirit; optimistic) |
| `intfloat__multilingual-e5-large-instruct_train_half_dev.json` | SVR `half_dev` (tuning only) |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_train_full_dev.json` | DeepSeek-R1 FP16 SVR **`full_dev_on_dev`** (Table 3) |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_test_train_full_dev.json` | DeepSeek-R1 FP16 SVR → **test** |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_test_train_full_dev_submission.csv` | matching submission |

## Paper-adjacent but incomplete

| File | Note |
|---|---|
| `encoder_ensemble_multilingual-e5-large-instruct+multilingual-e5-large_test_train_full_dev.json` | Weak stand-in for Table 4 **Encoders** (two e5 only, not paper LLMs) |
| `deepseek_feasibility_a1.json` | Tokenizer fix + 4-bit smoke; **not** Table 1 |
| `deepseek_r1_8b_nf4_subset_half_dev.json` | Quantized subset SVR; **not** Table 1/3 |
| `deepseek_feasibility.json`, `deepseek_4bit_probe.json`, `deepseek_r1_8b_feasibility.json`, `taide_feasibility.json` | GPU / load probes only |

## Non-paper / scratch (do not treat as reproduction claims)

| Pattern / file | Why |
|---|---|
| `*_domain_adapt*` | A4 labeled_dev weighting / retrieval / pseudo — **not in the paper** |
| `*_l2.json`, `*_tune.json` | Failed or flat ablations |
| `*_train_half_dev.json` for boost/ResNet | Intermediate tuning |
| `*_lgbm_train.json`, `*_xgb_train.json` (no strategy suffix) | Early runs |

Headline numbers for the paper path:

- e5-instruct SVR test: **0.488 / 0.788** MAE_V / MAE_A  
- Models ensemble test: **0.473 / 0.774**  
- DeepSeek-R1 FP16 `full_dev_on_dev`: **0.453 / 0.862**; test: **0.517 / 0.799**
