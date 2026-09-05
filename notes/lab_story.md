# Lab story: how this reproduction unfolded

This note is a **chronology of decisions**. Final claims, full metric tables, and paper comparisons live in [`report.md`](report.md). Post-discussion review (process, fixes, paper vs ours, discoveries): [`retrospective.md`](retrospective.md). DeepSeek Table 1–2 cells: [`table1_2_alignment.md`](table1_2_alignment.md). Cite JSON via [`results/README.md`](../results/README.md).

Hardware constraint that shaped every bet: **RTX 3070 8GB** for day-to-day work; **RTX 4090 24GB (RunPod)** later for paper LLM embeds.

---

## Stage 1 — Data and a runnable pipe

**Believed:** Match the shared-task sizes first; use something that fits an RTX 3070 8GB.

**Did:** Merged CVAT folds 1–5 → train **2954** (skipped broken `CVAT_all_SD.csv`). Dev **994**, test **1541**. Embedded with `multilingual-e5-large-instruct` + Instruct prefix; SVR RBF C=10, ε=0.2.

**Learned:** The pipe runs end-to-end. Without a clear eval protocol, early scores are easy to misread.

**Next:** Treat train mixes and holdouts like the paper’s Table 1 spirit.

---

## Stage 2 — The protocol trap

**Believed:** `half_dev` (train + half labeled_dev, score the other half) was our main quality signal.

**Did:** e5-instruct + SVR on that split landed around **MAE_A ≈ 1.04**. Grids and tree models barely moved arousal.

**Learned:** That number is **not** comparable to paper Table 3 (~0.81 on_dev) or the board (~0.76 on **test**). Mixing columns made the project look further behind than it was.

**Next:** Score held-out **test** with the official scorer; keep `half_dev` for tuning only.

---

## Stage 3 — A real test claim

**Believed:** `train_full_dev` → test matches how the shared task is judged.

**Did:** Official `scoring.py` on e5-instruct + SVR → about **0.488 / 0.788** MAE_V / MAE_A.

**Learned:** Test arousal was much kinder than half_dev suggested. Valence was already in a usable range vs top teams.

**Next:** Add the paper’s regressor zoo and average them (Table 4 Models spirit).

---

## Stage 4 — Table 4 Models

**Believed:** Averaging SVR with trees and a paper-style ResNet would beat single SVR without new embeds.

**Did:** CustomResNet + LGBM/XGB/CatBoost; five-way mean on test → about **0.473 / 0.774**.

**Learned:** Models ensemble beat the paper’s Models row and closed most of the gap to the board—but not quite the encoder-ensemble arousal (~0.76).

**Next:** Try another encoder that still fits an RTX 3070 8GB; sketch multi-encoder averaging.

---

## Stage 5 — More encoders without 8B VRAM

**Believed:** e5-large (no instruct) should track the paper’s Table 3 row; averaging two e5s might mimic “Encoders.”

**Did:** e5-large test ≈ **0.555 / 0.806** (near the paper’s e5-large line, weaker than instruct). Dual-e5 mean helped arousal a little and hurt valence.

**Learned:** Weak encoders drag the average. A real Table 4 Encoders claim needs the paper’s strong LLMs, not two e5s.

**Next:** Move LLM FP16 embedding to an **RTX 4090 24GB (RunPod)**; keep e5 caches for the final mix.

---

## Stage 6 — RTX 4090 24GB (RunPod) LLM path (Tables 1–3)

**Believed:** DeepSeek-R1 / Prover / TAIDE FP16 mean-pool + SVR would fill Table 3 and let us check Table 1–2.

**Did:** Full-corpus embeds via `embed_llm_full` (Qwen2 tokenizer fix for DeepSeek-R1). Table 1–2 cells filled; half_dev arousal stayed high (~1.05 vs paper ~0.81). `full_dev_on_dev` looked stronger but is optimistic.

**Learned:** Trends match the paper (labeled_dev helps; SVR beats trees on half_dev). Exact Table 1–2 cells still diverge—likely unpublished pooling/prompt details. Details: [`table1_2_alignment.md`](table1_2_alignment.md).

**Next:** Average all five encoder SVRs on **test**.

---

## Stage 7 — Table 4 Encoders and a lean repo

**Believed:** R1 + Prover + TAIDE + both e5s should approach the board.

**Did:** Five-encoder mean on test → about **0.470 / 0.758** (paper Encoders ~0.463 / 0.759; board ~0.46 / 0.76). Dropped side branches (domain adapt, probes, feasibility scratch) so the repo reads as paper-path only.

**Learned:** Headline shared-task reproduction is essentially done. Remaining gap is fidelity on optimistic/dev protocols and unpublished encoder details—not missing pipeline stages.

**Cite finals:** [`report.md`](report.md) §3 · [`results/README.md`](../results/README.md).
