# ROCLING-2025 TCU DSA reproduction (short weekly project)

This repo tries to reproduce parts of the paper  
*TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis*  
([ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)).

I ran everything on an **RTX 3070 (8GB)**. The main line is `multilingual-e5-large-instruct` + SVR. I did **not** run the full five-encoder setup or DeepSeek in full FP16.

**Write-up:** [`notes/report.md`](notes/report.md)  
**Day notes:** [`day3`](notes/day3.md) · [`day4`](notes/day4.md) · [`day6`](notes/day6.md) · [`day7`](notes/day7.md)  
**8B GPU notes:** [`feasibility_llm_8b.md`](notes/feasibility_llm_8b.md)

## Main test scores

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| e5-instruct + SVR | 0.488 | 0.788 | 0.788 | 0.578 |
| e5-large (no instruct) + SVR | 0.555 | 0.806 | 0.739 | 0.534 |
| Average of the two e5 SVRs | 0.499 | 0.774 | 0.784 | 0.585 |
| **Average of five regressors** (+CatBoost) | **0.473** | **0.774** | **0.795** | **0.598** |
| Paper Table 4 (Models) | 0.495 | 0.802 | 0.772 | 0.544 |
| TCU leaderboard (encoder ensemble) | 0.46 | 0.76 | 0.81 | 0.61 |

Half-dev check (instruct + SVR): MAE_V 0.541 / MAE_A 1.044. More detail in the day notes.

## Data

Shared-task clone: `data/raw/ROCLING-2025-ST-DSA-MST/`

- Train: merge `CVAT_1_SD.csv` … `CVAT_5_SD.csv` → **2954** rows (avoid `CVAT_all_SD.csv`; it has broken rows)
- Dev (with labels): `DSAMST-ValidationSet_ans.csv` → **994**
- Test (with labels): `DSAMST-TestSet_ans.csv` → **1541**
- Official scorer: `scoring.py`

Raw data and embedding `.npy` files are **not** in git (see `.gitignore`).

## How to run (Windows)

```powershell
cd D:\Projects\rocling-dsa-repro
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.train_resnet --strategy train_half_dev
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
```
