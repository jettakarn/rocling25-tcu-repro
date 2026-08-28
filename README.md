# 📊 Reproducing TCU at ROCLING-2025

Independent reproduction of  
*TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis*  
([ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)).  
e5 runs locally on an **RTX 3070 (8GB)**; LLM FP16 full-corpus embeds need a **≥16GB** GPU. Scores are comparable to TCU’s published board figures (no ordinal rank claimed).

[繁體中文](README.zh-TW.md) · English

## Docs

- Write-up: [`notes/report.md`](notes/report.md)
- Lab story (chronology): [`notes/lab_story.md`](notes/lab_story.md)
- Which result files to cite: [`results/README.md`](results/README.md)
- Table 1–2 alignment: [`notes/table1_2_alignment.md`](notes/table1_2_alignment.md)
- Paper: [ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)

## Setup

### 1. Environment

```powershell
cd D:\Projects\rocling-dsa-repro
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Dependencies

```powershell
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### 3. Run (e5 on 8GB)

```powershell
python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
```

### 4. LLM encoders (≥16GB)

FP16 embeds for DeepSeek-R1 / Prover / TAIDE and Table 3–4 runs (e.g. RunPod):

```bash
bash scripts/runpod_table3_encoders.sh
```

## Paper path

- [x] data → embed → SVR (e5-instruct + e5-large)
- [x] Table 4 Models — 5-regressor mean on test
- [x] Table 3 LLM encoders — DeepSeek-R1 / Prover / TAIDE FP16 (`embed_llm_full`)
- [x] Table 4 Encoders — 5-encoder mean on test (~board)
- [x] Tables 1–2 DeepSeek mixes — done; half_dev arousal still high vs paper  
  → details in [`notes/table1_2_alignment.md`](notes/table1_2_alignment.md)

## Evaluation protocols

- **`half_dev`** — train on train + half labeled_dev; score the other half (497). Tuning only.
- **`full_dev_on_dev`** — train on train + full labeled_dev; score_dev (994). Table 3–style; optimistic.
- **`test`** — train on train + full labeled_dev; score held-out test (1541). **Main claim** / board spirit.

**Leaderboard:** TCU’s board (~0.46 / 0.76 MAE) is closer to paper **Table 4 Encoders** on **`test`**, not **Table 3**. See [`notes/report.md`](notes/report.md).

## Headline results (paper path)

| Setup | Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| e5-instruct + SVR | test | 0.488 | 0.788 | 0.788 | 0.578 |
| Table 4 Models (5 regressors) | test | 0.473 | 0.774 | 0.795 | 0.598 |
| **5-encoder mean (Table 4 Encoders)** | **test** | **0.470** | **0.758** | **0.799** | **0.610** |
| Paper Table 4 Encoders | — | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU board | test | 0.46 | 0.76 | 0.81 | 0.61 |

## Single encoders (SVR, test)

| Encoder | MAE_V | MAE_A |
|---|---:|---:|
| e5-instruct | 0.488 | 0.788 |
| e5-large | 0.555 | 0.806 |
| DeepSeek-R1 FP16 | 0.517 | 0.799 |
| DeepSeek-Prover FP16 | 0.528 | 0.792 |
| TAIDE FP16 | 0.562 | 0.822 |

Optimistic `full_dev_on_dev` numbers live in [`notes/report.md`](notes/report.md). Prefer **test** for fair comparison to the board.

## Data

Shared-task clone: `data/raw/ROCLING-2025-ST-DSA-MST/`

- Train: merge `CVAT_1_SD.csv` … `CVAT_5_SD.csv` → **2954** rows (avoid `CVAT_all_SD.csv`; it has broken rows)
- Dev (with labels): `DSAMST-ValidationSet_ans.csv` → **994**
- Test (with labels): `DSAMST-TestSet_ans.csv` → **1541**
- Official scorer: `scoring.py`

Raw data and embedding `.npy` files are **not** in git (see `.gitignore`).

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
