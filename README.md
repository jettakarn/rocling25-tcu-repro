# ROCLING-2025 TCU Reproduction

This repo tries to reproduce parts of the paper  
*TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis*  
([ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)).

Local work runs on an **RTX 3070 (8GB)** for e5. LLM FP16 full-corpus embeds need a **≥16GB** GPU (e.g. RunPod); see `scripts/`.

**Write-up:** [`notes/report.md`](notes/report.md)  
**Which result files to cite:** [`results/README.md`](results/README.md)  
**Day notes:** [`day3`](notes/day3.md) · [`day4`](notes/day4.md) · [`day6`](notes/day6.md) · [`day7`](notes/day7.md)  
**Table 1–2 alignment:** [`table1_2_alignment.md`](notes/table1_2_alignment.md)  
**Cleanup notes:** [`cleanup_candidates.md`](notes/cleanup_candidates.md)  
**8B GPU notes:** [`feasibility_llm_8b.md`](notes/feasibility_llm_8b.md)

## Paper path

| Track | In this repo | Status |
|---|---|---|
| **Paper:** data → embed → SVR | e5-instruct (+ e5-large) | **done** |
| **Paper:** Table 4 Models (regressor mean) | SVR+LGBM+XGB+CatBoost+ResNet | **done** |
| **Paper:** Table 3 LLM encoders (DeepSeek-R1 / Prover / TAIDE FP16) | `embed_llm_full` + cloud | **done** |
| **Paper:** Table 4 Encoders (5-encoder mean on test) | R1+Prover+TAIDE+e5+e5-instruct | **done** (~board) |
| **Paper:** Tables 1–2 (DeepSeek mixes / regressors) | [`notes/table1_2_alignment.md`](notes/table1_2_alignment.md) | **done** (numbers diverge on half_dev A) |

## Evaluation protocols

| Protocol | What it means |
|---|---|
| **`half_dev`** | Train on train + half of labeled_dev; score the other half (497). For tuning only. |
| **`full_dev_on_dev`** | Train on train + full labeled_dev; score_dev (994). Matches paper **Table 3** spirit (optimistic). |
| **`test`** | Train on train + full labeled_dev; score held-out test (1541). **Main claim**; same spirit as the shared-task board. |

**Leaderboard:** TCU’s board (~0.46 / 0.76 MAE) is closer to paper **Table 4 Encoders** on **`test`**, not **Table 3**. See [`notes/report.md`](notes/report.md).

## Headline results (paper path)

| Setup | Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| e5-instruct + SVR | test | 0.488 | 0.788 | 0.788 | 0.578 |
| Table 4 Models (5 regressors) | test | 0.473 | 0.774 | 0.795 | 0.598 |
| **5-encoder mean (Table 4 Encoders)** | **test** | **0.470** | **0.758** | **0.799** | **0.610** |
| Paper Table 4 Encoders | — | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU board | test | 0.46 | 0.76 | 0.81 | 0.61 |

## Table 3–style single encoders (SVR)

| Encoder | `full_dev_on_dev` MAE_V / MAE_A | test MAE_V / MAE_A |
|---|---|---|
| e5-instruct | 0.497 / 0.984 | 0.488 / 0.788 |
| e5-large | — | 0.555 / 0.806 |
| DeepSeek-R1 FP16 | 0.453 / 0.862 | 0.517 / 0.799 |
| DeepSeek-Prover FP16 | 0.415 / 0.787 | 0.528 / 0.792 |
| TAIDE FP16 | 0.218 / 0.395 | 0.562 / 0.822 |

`full_dev_on_dev` is optimistic (dev labels were in training). Prefer **test** for fair comparison to the board.

Paper Table 4 Models reference: 0.495 / 0.802.

## 8GB note

- Optional NF4 probe scripts remain (`probe_llm`, `embed_llm`); not Table 1/3 claims. See [`feasibility_llm_8b.md`](notes/feasibility_llm_8b.md).

## Data

Shared-task clone: `data/raw/ROCLING-2025-ST-DSA-MST/`

- Train: merge `CVAT_1_SD.csv` … `CVAT_5_SD.csv` → **2954** rows (avoid `CVAT_all_SD.csv`; it has broken rows)
- Dev (with labels): `DSAMST-ValidationSet_ans.csv` → **994**
- Test (with labels): `DSAMST-TestSet_ans.csv` → **1541**
- Official scorer: `scoring.py`

Raw data and embedding `.npy` files are **not** in git (see `.gitignore`).

## How to run on Windows

```powershell
cd D:\Projects\rocling-dsa-repro
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Paper path
python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring

# Optional 8GB probe (not Table 1)
python -m src.probe_llm --model deepseek --download --load-in-4bit

# Cloud / ≥16GB: FP16 full corpus (Table 3–4)
# bash scripts/runpod_table3_encoders.sh
```

## Disclaimer

This repository provides an **independent reproduction** of the methodology described in the paper and is **not an official release** by the original authors.

The original authors did not release their code publicly. All implementation details, pipeline scripts, and experiments here are reconstructed independently based on the methodology outlined in the paper for academic research and reproducibility purposes.

## Citation & Acknowledgements

If you use this reproduction or reference the original methodology in your research, please cite the original paper:

```bibtex
@inproceedings{li-lin-2025-tcu,
  title     = {TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis},
  author    = {Li, Hsin-Chieh and Lin, Wen-Cheng},
  booktitle = {Proceedings of the 37th Conference on Computational Linguistics and Speech Processing (ROCLING 2025)},
  pages     = {399--406},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```
