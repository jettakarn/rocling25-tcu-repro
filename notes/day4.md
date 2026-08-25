# Day 4 lab notes

Same main line: `e5-instruct` + SVR (C=10, ε=0.2).  
Today: run on **test** and check with the official `scoring.py`.

## Data

| Split | n | Source |
|---|---:|---|
| train | 2954 | CVAT 5 folds |
| dev | 994 | DSAMST validation (labeled) |
| test | 1541 | `DSAMST-TestSet_ans.csv` → `data/processed/test.csv` |

`test.npy` shape `(1541, 1024)` with Instruct prefix and max_length=512.

## Commands

```powershell
python -m src.prepare_data
python -m src.embed --split test
python -m src.predict_test --strategy train_full_dev --run-official-scoring
```

## Test scores (same as official scoring)

| Train mix | MAE_V | MAE_A | PCC_V | PCC_A | n_train |
|---|---:|---:|---:|---:|---:|
| train only | 0.496 | 0.854 | 0.770 | 0.527 | 2954 |
| **train + full_dev** | **0.488** | **0.788** | **0.788** | **0.578** | 3948 |

Outputs:
- `results/…_test_train_full_dev_submission.csv`
- `results/…_test_train_full_dev.json`

Predictions clipped to [1, 9].

## Compare

| System | Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| This run | e5 + SVR, train+dev → test | **0.488** | **0.788** | **0.788** | **0.578** |
| Days 2–3 | e5 + SVR, half_dev holdout | 0.541 | 1.044 | 0.756 | 0.526 |
| Paper Table 3 | e5 + SVR, full_dev on_dev | 0.523 | 0.807 | 0.742 | 0.539 |
| TCU board | multi-encoder → test | 0.46 | 0.76 | 0.81 | 0.61 |
| CYUT-NLP (1st) | → test | 0.46 | 0.74 | 0.78 | 0.63 |

## Takeaways

- Test pipeline works; official scores match local metrics.
- Half-dev arousal **1.044 looked too harsh** — same model on test gets **0.788**, close to paper e5 (0.807) and not far from TCU’s 0.76.
- Valence on test (0.488) is better than half_dev (0.541), about 0.03 behind the top teams.
- Single e5+SVR is already good enough to report; averaging models is optional next.

## Skipped on purpose

- Averaging SVR+LGBM+XGB (low priority then)
- DeepSeek / five encoders

## Next

Day 5 write-up: [`report.md`](report.md).
