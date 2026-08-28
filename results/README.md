# Results: which files to trust

Official scores live under `results/`. Prefer the **canonical** list.

## Canonical (paper path — cite these)

| File | What it is |
|---|---|
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev.json` | e5-instruct + SVR → **test** |
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev_ensemble.json` | Table 4 **Models** |
| `intfloat__multilingual-e5-large_test_train_full_dev.json` | e5-large → test |
| `intfloat__multilingual-e5-large-instruct_train_full_dev.json` | e5-instruct `full_dev_on_dev` |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_{train,train_half_dev,train_full_dev}.json` | Table 1 mixes (SVR) |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_{lgbm,xgb,catboost}_train_half_dev.json` | Table 2 regressors |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_test_train_full_dev.json` | DeepSeek-R1 → test |
| `deepseek-ai__DeepSeek-Prover-V1.5-RL_{train,test_train}_full_dev.json` | Prover FP16 Table 3 |
| `taide__Llama3-TAIDE-LX-8B-Chat-Alpha1_{train,test_train}_full_dev.json` | TAIDE FP16 Table 3 |
| `encoder_ensemble_DeepSeek-R1-0528-Qwen3-8B+DeepSeek-Prover-V1.5-RL+Llama3-TAIDE-LX-8B-Chat-Alpha1+multilingual-e5-large+multilingual-e5-large-instruct_test_train_full_dev.json` | **Table 4 Encoders** |

Matching `*_submission.csv` files go with the test JSONs above.

## Kept probes (optional)

| File | Note |
|---|---|
| `deepseek_feasibility_a1.json` | Tokenizer fix history |
| `deepseek_r1_8b_feasibility.json`, `taide_feasibility.json` | Early GPU checks |

## Removed in high-priority cleanup

A4 / domain_adapt, `*_l2.json`, `*_tune.json`, NF4 subset / early `deepseek_feasibility.json` / `deepseek_4bit_probe.json`, dual-e5 encoder ensemble, strategy-less `*_lgbm_train.json` / `*_xgb_train.json`, `scripts/remote_diag_prover_tok.sh`. See [`notes/cleanup_candidates.md`](../notes/cleanup_candidates.md).

## Headline numbers

| System | Protocol | MAE_V | MAE_A |
|---|---|---:|---:|
| e5-instruct SVR | test | 0.488 | 0.788 |
| Models ensemble | test | 0.473 | 0.774 |
| **5-encoder ensemble** | **test** | **0.470** | **0.758** |
| Paper / board Encoders | test | ~0.46 | ~0.76 |
| DeepSeek Table 1 half_dev | half_dev | 0.578 | 1.047 |
| Paper Table 1 half_dev | half_dev | 0.524 | 0.809 |
