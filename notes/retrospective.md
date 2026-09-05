# Post-discussion retrospective

This note is a **review after a first discussion**: what we did, what broke, how we differ from the paper, what we learned, and what to improve next. It is **not** a second scoreboard.

Final claims and full metric tables: [`report.md`](report.md).  
Stage-by-stage chronology: [`lab_story.md`](lab_story.md).  
DeepSeek Table 1–2 cells: [`table1_2_alignment.md`](table1_2_alignment.md).  
Which JSON to cite: [`results/README.md`](../results/README.md).

Paper: [Li & Lin, ROCLING 2025](https://aclanthology.org/2025.rocling-main.44/).

---

## 1. What we did to get these results

Hardware that shaped every step: **RTX 3070 8GB** for day-to-day e5 work; **RTX 4090 24GB (RunPod)** later for paper LLM FP16 embeds.

### Data and scoring

- Merged shared-task CVAT folds `CVAT_1_SD` … `CVAT_5_SD` → train **2954** (skipped broken `CVAT_all_SD.csv`).
- Dev **994**, test **1541**, with labels for local evaluation.
- Scored with the official `scoring.py` (predictions clipped to [1, 9]).

### Local e5 path (RTX 3070 8GB)

- Embedded with `intfloat/multilingual-e5-large-instruct` (Instruct prefix, `max_length=512`) and later `e5-large` without instruct.
- Regressed valence / arousal with **SVR** (RBF, C=10, ε=0.2), matching the paper’s stated SVR setup.
- Named three evaluation protocols so we would stop mixing columns:
  - **`half_dev`** — train on train + half labeled_dev; score the other half (497). Tuning only.
  - **`full_dev_on_dev`** — train on train + full labeled_dev; score full dev (994). Table 3–style; optimistic.
  - **`test`** — train on train + full labeled_dev; score held-out test (1541). **Main claim** / board spirit.
- Headline single-encoder result (e5-instruct + SVR, `test`): about **0.488 / 0.788** MAE_V / MAE_A.

### Regressor zoo and Table 4 Models

- Added LightGBM, XGBoost, CatBoost, and a paper-style CustomResNet.
- Equal-weight **five-regressor mean** on test (Table 4 Models spirit): about **0.473 / 0.774**.
- Beat the paper’s Models row; still short of encoder-ensemble arousal (~0.76).

### Extra encoder tries on RTX 3070 8GB

- Ran e5-large alone (weaker than instruct; near the paper’s e5-large spirit on test).
- Tried a **dual-e5** average: arousal improved a little, valence got worse — weak encoders drag an equal-weight mix.

### RTX 4090 24GB (RunPod) LLM path (Tables 1–3)

- Built FP16 full-corpus embeds for **DeepSeek-R1**, **DeepSeek-Prover**, and **TAIDE** via `embed_llm_full` + shared helpers in `llm_encode` (including a Qwen2 tokenizer preset for DeepSeek-R1).
- Filled Table 3–style single-encoder rows and DeepSeek Table 1–2 train-mix / regressor cells (see `table1_2_alignment.md`).
- Trends matched the paper (labeled_dev helps; SVR beats trees on half_dev), but half_dev arousal stayed high vs paper cells.

### Table 4 Encoders and repo lean-up

- Averaged five encoder SVRs on **test** (R1 + Prover + TAIDE + e5 + e5-instruct) → about **0.470 / 0.758** (paper Encoders ~0.463 / 0.759; TCU board ~0.46 / 0.76).
- Dropped side branches from the paper-path view (probes, feasibility scratch, domain-adapt scripts) so the repo reads as reproduction-first.

**Method progress call:** about **95%**. Remaining gap is mostly unpublished pooling / prompt details, not missing pipeline stages.

---

## 2. Problems we hit and how we fixed them

| Problem | Symptom | Fix | Lesson |
|---|---|---|---|
| Broken all-in-one train CSV | `CVAT_all_SD.csv` had bad rows / wrong size | Merge folds 1–5 only | Never trust the convenience file; match paper sizes first |
| Protocol mix-up | half_dev MAE_A ~**1.04** looked far from board ~**0.76** | Name `half_dev` / `full_dev_on_dev` / `test`; claim on **test** | Wrong column makes a good run look like a failure |
| DeepSeek Chinese tokenization | Empty `input_ids` on Chinese; English OK | Use **Qwen2** tokenizer preset for DeepSeek-R1 | Model card / auto tokenizer choice can silently break CJK |
| RTX 3070 8GB cannot hold FP16 8B embeds | Weights alone ~16GB | Move LLM embeds to **RTX 4090 24GB (RunPod)**; keep e5 local | Hardware split is part of the method story |
| Fake “encoder ensemble” with two e5s | Dual-e5 hurt valence | Bring in real paper LLMs before averaging | Weak heads drag equal-weight means |
| Regressor chasing on arousal | SVR grid / trees barely moved half_dev A | Stop treating arousal as a hyperparameter problem | Bottleneck is embedding / domain, not the head |
| Table 1–2 cells still off after DeepSeek | half_dev MAE_A ~**1.05** vs paper ~**0.81** | Document gap; do not overclaim exact cells | Unpublished pooling / prompt / seed details limit fidelity |
| Side branches cluttering the repo | Probes, feasibility notes, domain_adapt mixed with claims | Strip non-paper path; keep chronology in `lab_story` | A lean paper-path repo is easier to review |

---

## 3. Original paper vs our reproduction

### Same spirit

- Pipeline: text → encoder embedding → valence / arousal regression.
- SVR settings stated in the paper (RBF, C=10, ε=0.2).
- Data sizes: train 2954 / dev 994 / test 1541.
- Tracks we covered: Table 3–style single encoders, Table 4 Models, Table 4 Encoders, DeepSeek Table 1–2 mixes.

### How we differ

| Topic | Paper | Ours |
|---|---|---|
| Code | Not released publicly | Independent reconstruction |
| Embedding recipe | Not fully specified (pooling, prompt, seed, …) | Assumed **mean-pool**; e5 Instruct prefix; LLM FP16 helpers as implemented |
| Headline board comparison | Table 4 Encoders / shared-task board | Five-encoder **test** mean **0.470 / 0.758** ≈ paper ~0.463 / 0.759 and board ~0.46 / 0.76 |
| Table 1–2 half_dev arousal | DeepSeek ~**0.81** MAE_A | DeepSeek-R1 FP16 ~**1.05** MAE_A (trend OK, cell not tight) |
| Protocol naming | Tables imply different train/eval mixes | We made three named protocols explicit so columns are not mixed |
| Hardware story | Not the focus of the write-up | RTX 3070 8GB local e5 + RTX 4090 24GB (RunPod) LLM embeds is how we actually ran it |
| Domain adapt / probes | Not part of the published paper path | Tried (e.g. labeled_dev upweighting), then **stripped** from the lean repo so claims stay paper-first |

### Where we match closely vs where we diverge

- **Close:** Table 4 Encoders / board on **`test`**; Models ensemble can beat the paper Models row; labeled_dev helps; SVR beats trees on half_dev.
- **Diverge:** DeepSeek Table 1–2 half_dev arousal; some optimistic `full_dev_on_dev` cells (especially strong-looking TAIDE on-dev numbers — do not use those as the main claim).

We treat the reproduction as **methodologically ~95%**: stages are filled; exact unpublished embedding details still block cell-level identity.

---

## 4. What we discovered

1. **Protocol choice changes the story more than small model tweaks.** Mixing half_dev with board/test numbers is the fastest way to misread progress.
2. **Arousal is the hard axis.** Changing SVR / trees barely helps; embedding quality and domain matter more.
3. **Model ensemble ≠ encoder ensemble.** Averaging regressors on one e5 can beat paper Models; board-level arousal needs multiple strong encoders.
4. **Weak encoders drag equal-weight averages.** Dual-e5 was a useful negative result before paying for LLM embeds.
5. **`full_dev_on_dev` is optimistic.** Dev labels were in training; TAIDE can look extremely strong there and still be ordinary on test.
6. **Labeled_dev helps domain shift.** Train CVAT text and medical-style dev/test text are not the same distribution; `train_full_dev` is not just “more rows.”
7. **Trends can reproduce without exact cells.** DeepSeek mixes and regressor rankings follow the paper’s spirit even when half_dev arousal does not match.

---

## 5. Improvement notes (for next round)

Priorities if we continue after this review — keep **paper reproduction** and **improvement track** labeled separately.

1. **Pooling / prompt ablations (paper fidelity).** Same DeepSeek weights; try last-token vs mean-pool, alternate Instruct/prefix strings, optional embedding L2 / `max_length`. Goal: move Table 1–2 half_dev MAE_A closer to ~0.81 without claiming a board win from that split.
2. **Weighted or filtered encoder mix (test / board).** Drop or down-weight weak heads; consider separate weights for V vs A. Goal: squeeze past equal-weight **0.470 / 0.758** on **test**.
3. **Domain adapt as an improvement track only.** Revisit labeled_dev upweighting / copies *after* the five-encoder mean, and report it as an extension — not as “the paper method.”

Do **not** spend the next round on more SVR grids for half_dev arousal, or on claiming wins from `full_dev_on_dev` alone.
