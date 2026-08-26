# ROCLING-2025 TCU Reproduction

This repo tries to reproduce parts of the paper  
*TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis*  
([ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)).

I ran everything on an **RTX 3070 (8GB)**. The main line is `multilingual-e5-large-instruct` + SVR. I did **not** run the full five-encoder setup or DeepSeek in full FP16.

**Write-up:** [`notes/report.md`](notes/report.md)  
**Day notes:** [`day3`](notes/day3.md) · [`day4`](notes/day4.md) · [`day6`](notes/day6.md) · [`day7`](notes/day7.md) · [`day8`](notes/day8.md)  
**8B GPU notes:** [`feasibility_llm_8b.md`](notes/feasibility_llm_8b.md)

## Evaluation protocols

| Protocol | What it means |
|---|---|
| **`half_dev`** | Train on train + half of labeled_dev; score the other half (497). For tuning only. |
| **`full_dev_on_dev`** | Train on train + full labeled_dev; score_dev (994). Matches paper **Table 3** spirit (optimistic). |
| **`test`** | Train on train + full labeled_dev; score held-out test (1541). **Main claim**; same spirit as the shared-task board. |

**Leaderboard:** TCU’s board (~0.46 / 0.76 MAE) is closer to paper **Table 4 Encoders** on **`test`**, not **Table 3** (single encoder on_dev). See [`notes/report.md`](notes/report.md) §2.4 and §3.1.

## Main scores (e5-instruct + SVR)

| Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| **`half_dev`** | 0.541 | 1.044 | 0.756 | 0.526 |
| **`full_dev_on_dev`** | 0.497 | 0.984 | 0.788 | 0.590 |
| **`test`** | **0.488** | **0.788** | **0.788** | **0.578** |

## Test benchmarks (`train_full_dev` → test)

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| e5-instruct + SVR | 0.488 | 0.788 | 0.788 | 0.578 |
| e5-large (no instruct) + SVR | 0.555 | 0.806 | 0.739 | 0.534 |
| Average of the two e5 SVRs | 0.499 | 0.774 | 0.784 | 0.585 |
| Average of five regressors (+CatBoost) | 0.473 | 0.774 | 0.795 | 0.598 |
| **Stronger labeled_dev (×3 weight)** | 0.496 | **0.765** | 0.789 | 0.601 |
| Paper Table 4 (Models) | 0.495 | 0.802 | 0.772 | 0.544 |
| Paper Table 4 (Encoders) | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU leaderboard (encoder ensemble) | 0.46 | 0.76 | 0.81 | 0.61 |

**Best arousal on this GPU so far:** labeled_dev ×3 SVR (MAE_A **0.765**). Details in [`notes/day8.md`](notes/day8.md).

## DeepSeek on 8GB (exploratory, not paper Table 1)

- Chinese tokenization: force `tokenizer_type="qwen2"` (repo’s LlamaTokenizerFast path returned empty CJK ids). See [`src/probe_llm.py`](src/probe_llm.py).
- 4-bit NF4 load works (~4.6GB). Subset SVR (300 train + 200_dev) is **weaker** than e5 on the same rows — labeled `quantized=true`, not a Table 1 reproduction. See [`notes/feasibility_llm_8b.md`](notes/feasibility_llm_8b.md).

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


python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.train_resnet --strategy train_half_dev
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring

# Optional: upweight labeled_dev (best MAE_A so far)
python -m src.domain_adapt --dev-copies 3

# Optional: DeepSeek 4-bit probe / subset embed (needs bitsandbytes)
python -m src.probe_llm --model deepseek --download --load-in-4bit
python -m src.embed_llm --help
```

## Disclaimer

This repository provides an **independent reproduction** of the methodology described in the paper and is **not an official release** by the original authors.

The original authors did not release their code publicly. All implementation details, pipeline scripts, and experiments here are reconstructed independently based on the methodology outlined in the paper for academic research and reproducibility purposes.

---

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

## Disclaimer
This repository provides an **independent reproduction** of the methodology described in the paper and is **not an official release** by the original authors. 

The original authors did not release their code publicly. All implementation details, pipeline scripts, and experiments here are reconstructed independently based on the methodology outlined in the paper for academic research and reproducibility purposes.

---

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
