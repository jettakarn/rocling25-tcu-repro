# Results: which files to cite

## Canonical (paper path)

| File | What it is |
|---|---|
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev.json` | e5-instruct + SVR → **test** |
| `intfloat__multilingual-e5-large-instruct_test_train_full_dev_ensemble.json` | Table 4 **Models** |
| `intfloat__multilingual-e5-large_test_train_full_dev.json` | e5-large → test |
| `intfloat__multilingual-e5-large-instruct_{train,train_half_dev,train_full_dev}.json` | e5-instruct Table 1-style mixes |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_{train,train_half_dev,train_full_dev}.json` | Table 1 mixes (SVR) |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_{lgbm,xgb,catboost}_train_half_dev.json` | Table 2 regressors |
| `deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_test_train_full_dev.json` | DeepSeek-R1 → test |
| `deepseek-ai__DeepSeek-Prover-V1.5-RL_{train,test_train}_full_dev.json` | Prover FP16 |
| `taide__Llama3-TAIDE-LX-8B-Chat-Alpha1_{train,test_train}_full_dev.json` | TAIDE FP16 |
| `encoder_ensemble_DeepSeek-R1-0528-Qwen3-8B+DeepSeek-Prover-V1.5-RL+Llama3-TAIDE-LX-8B-Chat-Alpha1+multilingual-e5-large+multilingual-e5-large-instruct_test_train_full_dev.json` | **Table 4 Encoders** |

Matching `*_submission.csv` files go with the test JSONs above.

## Headline numbers

| System | Protocol | MAE_V | MAE_A |
|---|---|---:|---:|
| e5-instruct SVR | test | 0.488 | 0.788 |
| Models ensemble | test | 0.473 | 0.774 |
| **5-encoder ensemble** | **test** | **0.470** | **0.758** |
| Paper / board Encoders | test | ~0.46 | ~0.76 |
| DeepSeek Table 1 half_dev | half_dev | 0.578 | 1.047 |
| Paper Table 1 half_dev | half_dev | 0.524 | 0.809 |
