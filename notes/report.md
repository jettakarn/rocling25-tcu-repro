# Project report: ROCLING-2025 TCU DSA reproduction

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/)  
Machine: RTX 3070 8GB, Python 3.13, CUDA torch  
Main setup: **`intfloat/multilingual-e5-large-instruct` + SVR** (RBF, C=10, ε=0.2)

I did **not** fully reproduce DeepSeek FP16 embedding or the paper’s five-encoder mix.

Day notes: `day3.md`, `day4.md`, `day6.md`, `day7.md`, `day8.md`.  
**Which JSON to cite:** [`results/README.md`](../results/README.md).

### Paper path vs extras

| Track | What | Status |
|---|---|---|
| **Paper** | data → e5 embed → SVR / Table 3-style protocols | done |
| **Paper** | Table 4 Models (regressor mean) | done |
| **Paper** | DeepSeek + multi-LLM → Table 4 Encoders | **not done** (needs ≥16GB) |
| **Non-paper** | A4 `domain_adapt` (labeled_dev ×3, etc.) | optional only |

---

## 1. Goal and takeaways

Goal: rebuild the “embed text → regress valence/arousal” pipeline with something that fits in 8GB, match the paper’s data sizes, and score with the official script.

| Item | Track | Result |
|---|---|---|
| Data | paper | train 2954 (5 folds), dev 994, test 1541 |
| **e5-instruct + SVR (`half_dev`)** | paper | 0.541 / 1.044 / 0.756 / 0.526 |
| **e5-instruct + SVR (`full_dev_on_dev`)** | paper | 0.497 / 0.984 / 0.788 / 0.590 |
| **e5-instruct + SVR (`test`)** | paper | **0.488 / 0.788 / 0.788 / 0.578** |
| **Test (five-model average)** | paper (Table 4 Models) | **0.473 / 0.774 / 0.795 / 0.598** |
| e5-large (no instruct) on test | paper | 0.555 / 0.806 / 0.739 / 0.534 |
| A4 labeled_dev ×3 | **non-paper** | 0.496 / 0.765 / 0.789 / 0.601 |
| vs paper e5 (Table 3) | — | ours MAE_A 0.984 on_dev vs paper 0.807 |
| DeepSeek-R1 8B | paper gap | 4-bit probe only; FP16 Table 1/3 **not** done |
| Bottom line (paper path) | — | best paper-path valence **0.473**, arousal **0.774** (Models ensemble) |

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
- Predictions clipped to [1, 9]; matches official `scoring.py`

### 2.4 Evaluation protocols (read before comparing numbers)

We report three eval columns side by side. They answer different questions; do not mix them when comparing to the paper or the leaderboard.

| Protocol | Train set | Scored on | Role |
|---|---|---|---|
| **`half_dev`** | train + half of labeled dev (3451 rows) | other half of dev (497) | tuning / sanity check (noisy on arousal) |
| **`full_dev_on_dev`** | train + full labeled dev (3948 rows) | full dev (994) | matches paper **Table 3** spirit (optimistic) |
| **`test`** | train + full labeled dev (3948 rows) | held-out test (1541) | **main claim**; matches shared-task / board scoring |

**Leaderboard note:** TCU’s published board scores (~0.46 / 0.76 MAE) follow the spirit of paper **Table 4 Encoders** — multi-encoder ensemble on **test** — not **Table 3** (single-encoder SVR on **dev** after training on train+full_dev). Our headline numbers use **`test`** after `train_full_dev`.

### 2.5 Commands

```powershell
cd D:\Projects\rocling-dsa-repro
.\.venv\Scripts\Activate.ps1

python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
# non-paper only: python -m src.domain_adapt --dev-copies 3
```

---

## 3. Results

All metrics below for **e5-instruct + SVR** unless noted. Source: `results/*.json`.

### 3.1 Side-by-side: `half_dev` / `full_dev_on_dev` / `test`

| Setup | Protocol | n_train | n_eval | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|---:|---:|
| train only | dev (not half split) | 2954 | 994 | 0.586 | 1.120 | 0.725 | 0.490 |
| **train_half_dev** | **`half_dev`** | 3451 | 497 | **0.541** | **1.044** | **0.756** | **0.526** |
| **train_full_dev** | **`full_dev_on_dev`** | 3948 | 994 | **0.497** | **0.984** | **0.788** | **0.590** |
| train_full_dev | **`test`** | 3948 | 1541 | **0.488** | **0.788** | **0.788** | **0.578** |

Paper **Table 3** e5-instruct row (SVR + train+full_dev, scored on dev): 0.523 / 0.807 / 0.742 / 0.539 — same **`full_dev_on_dev`** column as our 0.497 / 0.984 row, not our **`test`** row.

Small checks on **`half_dev`**: SVR grid best MAE_A 1.033; LGBM 1.051; XGB 1.046; L2 1.044. Changing the regressor did not really help arousal.

### 3.2 Test extras (all `train_full_dev` → `test`)

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| train only (SVR) | 0.496 | 0.854 | 0.770 | 0.527 |
| **e5-instruct + SVR** | **0.488** | **0.788** | **0.788** | **0.578** |
| e5-large (no instruct) + SVR | 0.555 | 0.806 | 0.739 | 0.534 |
| dual e5 SVR mean (encoder ensemble) | 0.499 | 0.774 | 0.784 | 0.585 |
| CustomResNet | 0.455 | 0.797 | 0.790 | 0.585 |
| Average SVR+LGBM+XGB+ResNet | 0.472 | 0.776 | 0.795 | 0.596 |
| **+ CatBoost (five models)** | **0.473** | **0.774** | **0.795** | **0.598** |
| A4 SVR labeled_dev ×3 (**non-paper**) | 0.496 | 0.765 | 0.789 | 0.601 |

### 3.3 Compare to the paper / leaderboard

| System | Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| This repo, e5+SVR | **`test`** | 0.488 | 0.788 | 0.788 | 0.578 |
| This repo, e5+SVR | **`full_dev_on_dev`** | 0.497 | 0.984 | 0.788 | 0.590 |
| This repo, five-model average | **`test`** | **0.473** | 0.774 | **0.795** | 0.598 |
| This repo, A4 (**non-paper**) | **`test`** | 0.496 | 0.765 | 0.789 | 0.601 |
| Paper Table 3 e5-instruct | **`full_dev_on_dev`** | 0.523 | 0.807 | 0.742 | 0.539 |
| Paper Table 4 Models | paper (mixed) | 0.495 | 0.802 | 0.772 | 0.544 |
| Paper Table 4 Encoders | paper (mixed) | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU multi-encoder | **`test`** (board) | 0.46 | 0.76 | 0.81 | 0.61 |
| CYUT-NLP (1st) | **`test`** | 0.46 | 0.74 | 0.78 | 0.63 |

**How to read this table**

- Compare our **`test`** rows to TCU / CYUT and to paper **Table 4 Encoders** — all are held-out **`test`** (or paper ensemble) claims, not Table 3 dev scores.
- Compare our **`full_dev_on_dev`** row to paper **Table 3** e5-instruct — both train on train+full_dev and score on **dev** (optimistic; dev labels were in training).
- Do **not** compare our **`half_dev`** arousal (1.044) to paper Table 3 (0.807) or the board (0.76); the holdout is small and a different split.
- Tables 1–2 in the paper mostly use **DeepSeek**, not e5.

### 3.4 DeepSeek / other 8B models

See [`feasibility_llm_8b.md`](feasibility_llm_8b.md).  
On 8GB, full FP16 embedding is not realistic. 4-bit NF4 loads (~4.6GB); Chinese tokenizer is fixed via `qwen2`. Subset SVR is exploratory only (`quantized=true`), not Tables 1–3.

---

## 4. Discussion

1. **Use the right protocol column.**  
   `half_dev` MAE_A was 1.044; the same SVR on **`test`** got 0.788. Paper Table 3 (0.807) belongs in **`full_dev_on_dev`**, not **`half_dev`** or **`test`**. The board (~0.76 MAE_A) belongs with **`test`** / Table 4 **Encoders**, not Table 3.

2. **Single e5 on test is already strong; averaging models helps a bit more.**  
   Test arousal 0.788 is closer to the board than half-dev tuning suggested. After five-model averaging we are about 0.01–0.02 MAE behind TCU. Closing that gap likely needs more encoders and more VRAM.

3. **Domain shift.**  
   Train is general CVAT text; dev/test are medical reflections. Adding labeled_dev into training (`train_full_dev`) helps on test. Day 8 **A4** (non-paper) further weights labeled_dev ×3 → MAE_A 0.765; that is an extra experiment, not a paper claim.

4. **What is still missing for a full paper reproduction.**  
   Full DeepSeek FP16 embedding and multi-LLM encoder averaging (Table 4 Encoders). 4-bit DeepSeek is only a feasibility check.

---

## 5. Limits

- 8GB GPU; embed batch size 16; no fine-tuning of the encoder
- Scoring **dev** after `train_full_dev` is optimistic; using that train setup for **test** is fine
- Half-dev split uses `seed=42`
- See `requirements.txt`; install CUDA torch separately if needed

---

## 6. What is in the repo

| Path | Track | What it is |
|---|---|---|
| `configs/experiment.yaml` | paper | seeds, encoder, SVR, paths |
| `src/prepare_data.py` | paper | builds train/dev/test csv |
| `src/embed.py` | paper | e5 embeddings |
| `src/train_svr.py` / `train_boost.py` | paper | regressors |
| `src/predict_test.py` | paper | test predict + official scoring |
| `src/custom_resnet.py` / `train_resnet.py` | paper | paper-style MLP |
| `src/ensemble_models.py` | paper | Table 4 Models average |
| `src/ensemble_encoders.py` | paper (partial) | multi-encoder SVR mean |
| `src/probe_llm.py` / `embed_llm.py` | paper gap / probe | 8B load + NF4 subset embed |
| `src/tune_svr.py` | scratch | SVR grid (flat) |
| `src/domain_adapt.py` | **non-paper** | A4 labeled_dev weight / retrieval / pseudo |
| `src/inspect_data.py` | util | CSV peek |
| `results/README.md` | — | which JSON files to cite |
| `notes/` | — | day notes and this report |

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
| 8 | A4 domain adapt (dev ×3 → MAE_A 0.765) | done |
