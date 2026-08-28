# Day 6 lab notes

Added **CustomResNet** and a **model average** (SVR + LGBM + XGB + ResNet).  
Also checked whether **DeepSeek-R1 8B** can run on this GPU (no full embedding yet).

## Commands

```powershell
python -m src.train_resnet --strategy train_half_dev
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
python -m src.probe_llm --model deepseek
```

## CustomResNet

Code: `src/custom_resnet.py`  
Roughly: RMSNorm → 6 residual blocks → two linear layers → `sin(x)*4+5` (maps to 1–9).  
Paper does not give the hidden size; I used `hidden=512`, Adam, MSE, 40 epochs.

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| half_dev | 0.666 | 1.046 | 0.742 | 0.510 |
| test (train+dev; one model inside the ensemble) | **0.455** | 0.797 | 0.790 | 0.585 |

Half-dev valence is weak; on test, ResNet has the best valence of the single models.

## Model average on test (official scoring)

| Member | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| SVR | 0.488 | 0.788 | 0.788 | 0.578 |
| LGBM | 0.491 | 0.791 | 0.788 | 0.569 |
| XGB | 0.496 | 0.781 | 0.784 | 0.575 |
| CustomResNet | 0.455 | 0.797 | 0.790 | 0.585 |
| **Mean** | **0.472** | **0.776** | **0.795** | **0.596** |

| System | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| Day 4 SVR only | 0.488 | 0.788 | 0.788 | 0.578 |
| **Day 6 average** | **0.472** | **0.776** | **0.795** | **0.596** |
| Paper Table 4 Models | 0.495 | 0.802 | 0.772 | 0.544 |
| Paper Table 4 Encoders | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU board | 0.46 | 0.76 | 0.81 | 0.61 |

Averaging beats single SVR and also beats the paper’s “Models” row. Still a bit behind the “Encoders” row / leaderboard (~0.02 MAE_A).

## DeepSeek-R1 8B

See [`feasibility_llm_8b.md`](feasibility_llm_8b.md).  
FP16 embedding does not fit. 4-bit might load; I did not make it a must-do for this week.

## Files

- `src/custom_resnet.py`, `train_resnet.py`, `ensemble_models.py`, `probe_llm.py`
- `results/*_ensemble*`
- `results/deepseek_r1_8b_feasibility.json`

## Day 7

More checks (e5-large, CatBoost, dual e5). No need for full DeepSeek embedding unless the tokenizer issue is fixed.
