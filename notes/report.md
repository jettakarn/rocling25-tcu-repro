# Project report: ROCLING-2025 TCU DSA reproduction

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/)  
Machines: RTX 3070 8GB (e5); RTX 4090 24GB (RunPod) for LLM FP16 embeds  
Main local setup: **`intfloat/multilingual-e5-large-instruct` + SVR** (RBF, C=10, ε=0.2)

Lab story: [`lab_story.md`](lab_story.md).  
Post-discussion review: [`retrospective.md`](retrospective.md).  
Table 1–2: [`table1_2_alignment.md`](table1_2_alignment.md).  
**Which JSON to cite:** [`results/README.md`](../results/README.md).

### Paper path

| Track | What | Status |
|---|---|---|
| **Paper** | data → e5 embed → SVR / Table 3-style protocols | **done** |
| **Paper** | Table 4 Models (regressor mean) | **done** |
| **Paper** | DeepSeek-R1 / Prover / TAIDE FP16 → SVR (Table 3) | **done** |
| **Paper** | Table 4 Encoders (5-encoder mean on test) | **done** (**0.470 / 0.758**) |
| **Paper** | Tables 1–2 DeepSeek train mixes / regressors | **done** (see `notes/table1_2_alignment.md`) |

**Reproduction progress (method):** about **95%**. Remaining gap is mostly unpublished pooling/prompt details (Table 1–2 half_dev arousal still higher than paper).


---

## 1. Goal and takeaways

Goal: rebuild embed → valence/arousal regression, match paper data sizes, score with the official script, and cover Table 3–4 spirit.

| Item | Track | Result |
|---|---|---|
| Data | paper | train 2954, dev 994, test 1541 |
| e5-instruct SVR test | paper | 0.488 / 0.788 / 0.788 / 0.578 |
| Table 4 Models (5 regressors) | paper | 0.473 / 0.774 / 0.795 / 0.598 |
| **5-encoder mean (Table 4 Encoders)** | paper | **0.470 / 0.758 / 0.799 / 0.610** |
| DeepSeek-R1 / Prover / TAIDE | paper Table 3 | see §3.4 |
| DeepSeek Table 1–2 cells | paper | see [`table1_2_alignment.md`](table1_2_alignment.md) |
| Paper / board Encoders | — | ~0.46 / 0.76 |
| Bottom line | — | Encoder ensemble ≈ board; Table 1–2 half_dev arousal still higher than paper |

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
- `max_length=512` (no embedding L2)
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

Small checks on **`half_dev`**: changing the regressor (grid / LGBM / XGB) did not really help arousal.

### 3.2 Test extras (all `train_full_dev` → `test`)

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| train only (SVR) | 0.496 | 0.854 | 0.770 | 0.527 |
| **e5-instruct + SVR** | **0.488** | **0.788** | **0.788** | **0.578** |
| e5-large (no instruct) + SVR | 0.555 | 0.806 | 0.739 | 0.534 |
| CustomResNet | 0.455 | 0.797 | 0.790 | 0.585 |
| Average SVR+LGBM+XGB+ResNet | 0.472 | 0.776 | 0.795 | 0.596 |
| **+ CatBoost (five models)** | **0.473** | **0.774** | **0.795** | **0.598** |
| **5-encoder mean (R1+Prover+TAIDE+e5+e5-instruct)** | **0.470** | **0.758** | **0.799** | **0.610** |

### 3.3 Compare to the paper / leaderboard

| System | Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| This repo, e5+SVR | **`test`** | 0.488 | 0.788 | 0.788 | 0.578 |
| This repo, five-model average | **`test`** | 0.473 | 0.774 | 0.795 | 0.598 |
| **This repo, 5-encoder mean** | **`test`** | **0.470** | **0.758** | **0.799** | **0.610** |
| Paper Table 3 e5-instruct | **`full_dev_on_dev`** | 0.523 | 0.807 | 0.742 | 0.539 |
| Paper Table 4 Models | paper (mixed) | 0.495 | 0.802 | 0.772 | 0.544 |
| Paper Table 4 Encoders | paper (mixed) | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU multi-encoder | **`test`** (board) | 0.46 | 0.76 | 0.81 | 0.61 |
| CYUT-NLP (1st) | **`test`** | 0.46 | 0.74 | 0.78 | 0.63 |

**How to read this table**

- Compare our **`test`** rows to TCU / CYUT and to paper **Table 4 Encoders**.
- Compare **`full_dev_on_dev`** to paper **Table 3** (optimistic).
- Do **not** compare **`half_dev`** arousal (1.044) to Table 3 or the board.

### 3.4 DeepSeek / Prover / TAIDE (FP16 full corpus)

RTX 4090 24GB (RunPod); `src/embed_llm_full.py` + `src/llm_encode.py`; `quantized=false`.
DeepSeek-R1 Chinese tokenization uses forced Qwen2 BPE (`tokenizer_type="qwen2"`).

| Encoder | `full_dev_on_dev` | test |
|---|---|---|
| DeepSeek-R1 | 0.453 / 0.862 / 0.829 / 0.669 | 0.517 / 0.799 / 0.743 / 0.555 |
| DeepSeek-Prover | 0.415 / 0.787 / 0.851 / 0.729 | 0.528 / 0.792 / 0.744 / 0.563 |
| TAIDE | 0.218 / 0.395 / 0.971 / 0.928 | 0.562 / 0.822 / 0.717 / 0.542 |

TAIDE `full_dev_on_dev` looks extremely strong (optimistic protocol). Prefer **test** for claims. Five-encoder average on test is the Table 4 Encoders claim (**0.470 / 0.758**).

---

## 4. Discussion

1. **Use the right protocol column.**  
   Board / Table 4 Encoders ↔ **`test`**. Table 3 ↔ **`full_dev_on_dev`**. Do not mix with `half_dev`.

2. **Encoder ensemble closes the board gap.**  
   Models-only average left ~0.01–0.02 MAE_A behind TCU. Five-encoder mean hits **0.758** MAE_A (≈ board 0.76 / paper 0.759); valence **0.470** vs board ~0.46.

3. **Domain shift.**  
   `train_full_dev` helps on medical-style test text.

4. **What remains.**  
   Table 1–2 DeepSeek cells are filled (`notes/table1_2_alignment.md`) but half_dev MAE_A does not match paper (~1.05 vs 0.81). Headline test path (Table 4 Encoders) is done.
---

## 5. Limits

- RTX 3070 8GB for e5; embed batch size 16; no fine-tuning of the encoder
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
| `src/ensemble_encoders.py` | paper | multi-encoder SVR mean (Table 4 Encoders) |
| `src/llm_encode.py` | paper | LLM tokenizer presets + mean-pool helpers |
| `src/embed_llm_full.py` | paper | FP16/BF16 full-corpus LLM embed (RTX 4090 24GB / RunPod) |
| `configs/deepseek_r1.yaml` / `deepseek_prover.yaml` / `taide.yaml` | paper | LLM encoder configs |
| `scripts/runpod_table3_encoders.sh` | paper | RTX 4090 24GB (RunPod) embed + SVR + encoder mix |
| `results/README.md` | — | which JSON files to cite |
| `notes/` | — | day notes and this report |

---

## 7. Chronology

Stage-by-stage decisions (not a second scoreboard): [`lab_story.md`](lab_story.md).
