# Project report: ROCLING-2025 TCU DSA reproduction

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/)  
Machine: RTX 3070 8GB, Python 3.13, CUDA torch  
Main setup: **`intfloat/multilingual-e5-large-instruct` + SVR** (RBF, C=10, ε=0.2)

I did **not** fully reproduce DeepSeek FP16 embedding or the paper’s five-encoder mix. Later days add CustomResNet, model averaging, e5-large, and some GPU checks for 8B models.

Day notes: `day3.md`, `day4.md`, `day6.md`, `day7.md`.  
Scores as JSON: `results/*.json`.

---

## 1. Goal and takeaways

Goal: rebuild the “embed text → regress valence/arousal” pipeline with something that fits in 8GB, match the paper’s data sizes, and score with the official script.

| Item | Result |
|---|---|
| Data | train 2954 (5 folds), dev 994, test 1541 |
| Half-dev (for tuning) | MAE_V 0.541 / MAE_A **1.044** / PCC_V 0.756 / PCC_A 0.526 |
| **Test (SVR)** | **0.488 / 0.788 / 0.788 / 0.578** |
| **Test (five-model average, final)** | **0.473 / 0.774 / 0.795 / 0.598** |
| e5-large (no instruct) on test | 0.555 / 0.806 / 0.739 / 0.534 (close to paper Table 3) |
| vs paper e5 (Table 3, full-dev eval) | our instruct test arousal ≈ 0.788 vs paper 0.807 |
| vs TCU leaderboard | about +0.01–0.02 MAE behind their big ensemble |
| DeepSeek-R1 8B | 4-bit load works (~4.6GB); Chinese tokenizer broke full embedding for now |
| Bottom line | half-dev arousal looks worse than test; best 8GB result is e5-instruct + model average |

---

## 2. Method

### 2.1 Data

| Split | n | Source |
|---|---:|---|
| train | 2954 | merge `CVAT_1_SD` … `CVAT_5_SD` (do **not** use `CVAT_all_SD.csv`) |
| dev | 994 | `DSAMST-ValidationSet_ans.csv` |
| test | 1541 | `DSAMST-TestSet_ans.csv` |

Length and VA means look like the paper (train ~57.6 chars, V/A ≈ 4.8; dev/test are longer medical reflection texts).

### 2.2 Embeddings

- Model: `multilingual-e5-large-instruct`
- Prefix: `Instruct: {task}\nQuery: {text}`
- `max_length=512`, no L2 by default (Day 3 tried L2; almost no change)
- Size: 1024-d vectors under `data/embeddings/…`

### 2.3 Regression

- SVR RBF, C=10, ε=0.2 (same as the paper)
- Train modes: `train` / `train_half_dev` / `train_full_dev`
- For test I train on **train + full_dev**, then score on held-out test
- Predictions clipped to [1, 9]; matches official `scoring.py`

### 2.4 Commands

```powershell
cd D:\Projects\rocling-dsa-repro
.\.venv\Scripts\Activate.ps1

python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
```

---

## 3. Results

### 3.1 Half-dev (e5 + SVR)

| Strategy | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| train | 0.586 | 1.120 | 0.725 | 0.490 |
| train_half_dev | **0.541** | **1.044** | **0.756** | **0.526** |
| train_full_dev (too optimistic on dev) | 0.497 | 0.984 | 0.788 | 0.590 |

Small checks on half-dev: SVR grid best MAE_A 1.033; LGBM 1.051; XGB 1.046; L2 1.044. Changing the regressor did not really help arousal.

### 3.2 Test

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| train only (SVR) | 0.496 | 0.854 | 0.770 | 0.527 |
| **train + full_dev (SVR)** | **0.488** | **0.788** | **0.788** | **0.578** |
| CustomResNet (same train) | 0.455 | 0.797 | 0.790 | 0.585 |
| Average SVR+LGBM+XGB+ResNet | 0.472 | 0.776 | 0.795 | 0.596 |
| **+ CatBoost (five models)** | **0.473** | **0.774** | **0.795** | **0.598** |

### 3.3 Compare to the paper / leaderboard

| System | Eval | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| This repo, e5+SVR | test | 0.488 | 0.788 | 0.788 | 0.578 |
| **This repo, model average** | test | **0.473** | **0.774** | **0.795** | **0.598** |
| Paper Table 3 e5-instruct | full_dev (dev) | 0.523 | 0.807 | 0.742 | 0.539 |
| Paper Table 4 Models | paper | 0.495 | 0.802 | 0.772 | 0.544 |
| Paper Table 4 Encoders | paper | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU multi-encoder | test (board) | 0.46 | 0.76 | 0.81 | 0.61 |
| CYUT-NLP (1st) | test | 0.46 | 0.74 | 0.78 | 0.63 |

Note: Tables 1–2 in the paper mostly use **DeepSeek**. Table 3 e5 numbers are on **dev with full_dev training**, not the same as our **test** numbers. Compare carefully.

### 3.4 DeepSeek / other 8B models

See [`feasibility_llm_8b.md`](feasibility_llm_8b.md).  
On 8GB, full FP16 embedding is not realistic. 4-bit can load, but Chinese tokenization still blocks a full run here.

---

## 4. Discussion

1. **Half-dev is not the same as test.**  
   Half-dev MAE_A was 1.044; the same idea on test got 0.788. The holdout is small and noisy, so I should not judge only from half-dev.

2. **Single e5 is already close to the paper’s e5; averaging models helps a bit more.**  
   Test arousal is in the same range as paper Table 3. After averaging, we are about 0.01–0.02 MAE behind the TCU board. Closing that gap likely needs stronger / more encoders and more VRAM.

3. **Domain shift.**  
   Train is general CVAT text; dev/test are medical reflections. Adding labeled_dev into training helps a lot on test (especially arousal).

4. **What I skipped.**  
   Full DeepSeek FP16 embedding and five big encoders. CustomResNet and model averaging are in the repo. 4-bit DeepSeek is only a feasibility check for now.

---

## 5. Limits

- 8GB GPU; embed batch size 16; no fine-tuning of the encoder
- Scoring **dev** after `train_full_dev` is optimistic; using that train setup for **test** is fine
- Half-dev split uses `seed=42`
- See `requirements.txt`; install CUDA torch separately if needed

---

## 6. What is in the repo

| Path | What it is |
|---|---|
| `configs/experiment.yaml` | seeds, encoder, SVR, paths |
| `src/prepare_data.py` | builds train/dev/test csv |
| `src/embed.py` | e5 embeddings |
| `src/train_svr.py` / `tune_svr.py` / `train_boost.py` | regressors |
| `src/predict_test.py` | test predict + official scoring |
| `src/custom_resnet.py` / `train_resnet.py` | paper-style MLP |
| `src/ensemble_models.py` | average several regressors |
| `src/ensemble_encoders.py` | average SVRs from more than one encoder |
| `src/probe_llm.py` | 8B load checks |
| `results/*` | metrics JSON and submission csv |
| `notes/` | day notes and this report |

---

## 7. Week overview

| Day | What I did | Status |
|---|---|---|
| 1 | Data; 2954 train rows | done |
| 2 | Embed + SVR / grid / trees | done |
| 3 | Tables + L2 try | done |
| 4 | Test + scoring | done |
| 5 | This report | done |
| 6 | ResNet, model average, DeepSeek check | done |
| 7 | e5-large, CatBoost, dual e5, 4-bit probe | done |
