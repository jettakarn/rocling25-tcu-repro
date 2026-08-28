# 📊 重現 TCU @ ROCLING-2025

獨立重現論文  
*TCU at ROCLING-2025 Shared Task: Leveraging LLM Embeddings and Ensemble Regression for Chinese Dimensional Sentiment Analysis*  
（[ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)）。  
e5 可在本機 **RTX 3070（8GB）** 跑；LLM FP16 全語料 embedding 需 **≥16GB** GPU。分數與 TCU 公開榜數字相當（不宣稱名次）。

[English](README.md) · 繁體中文

## 文件

- 報告：[`notes/report.md`](notes/report.md)
- 實驗時序（lab story）：[`notes/lab_story.md`](notes/lab_story.md)
- 應引用的結果檔：[`results/README.md`](results/README.md)
- Table 1–2 對齊說明：[`notes/table1_2_alignment.md`](notes/table1_2_alignment.md)
- 論文：[ACL Anthology](https://aclanthology.org/2025.rocling-main.44/)

## 環境設置

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

### 3. Run（e5，8GB）

```powershell
python -m src.prepare_data
python -m src.embed --split all
python -m src.train_svr --strategy train_half_dev
python -m src.predict_test --strategy train_full_dev --run-official-scoring
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
```

### 4. LLM encoders（≥16GB）

DeepSeek-R1 / Prover / TAIDE 的 FP16 embedding 與 Table 3–4（例如 RunPod）：

```bash
bash scripts/runpod_table3_encoders.sh
```

## Paper path

- [x] data → embed → SVR（e5-instruct + e5-large）
- [x] Table 4 Models — test 上五個 regressor 平均
- [x] Table 3 LLM encoders — DeepSeek-R1 / Prover / TAIDE FP16（`embed_llm_full`）
- [x] Table 4 Encoders — test 上五個 encoder 平均（≈榜）
- [x] Tables 1–2 DeepSeek mixes — 完成；half_dev arousal 仍高於論文  
  → 細節見 [`notes/table1_2_alignment.md`](notes/table1_2_alignment.md)

## Evaluation protocols

- **`half_dev`** — 用 train + 半份 labeled_dev 訓練；評另一半（497）。僅供調參。
- **`full_dev_on_dev`** — 用 train + 完整 labeled_dev 訓練；評 score_dev（994）。Table 3 風格；偏樂觀。
- **`test`** — 用 train + 完整 labeled_dev 訓練；評 held-out test（1541）。**主宣稱**／與榜精神一致。

**Leaderboard：** TCU 榜（約 0.46 / 0.76 MAE）較接近論文 **Table 4 Encoders** 在 **`test`** 上的設定，而非 **Table 3**。見 [`notes/report.md`](notes/report.md)。

## Headline results（paper path）

| Setup | Protocol | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---|---:|---:|---:|---:|
| e5-instruct + SVR | test | 0.488 | 0.788 | 0.788 | 0.578 |
| Table 4 Models（5 regressors） | test | 0.473 | 0.774 | 0.795 | 0.598 |
| **5-encoder mean（Table 4 Encoders）** | **test** | **0.470** | **0.758** | **0.799** | **0.610** |
| Paper Table 4 Encoders | — | 0.463 | 0.759 | 0.805 | 0.608 |
| TCU board | test | 0.46 | 0.76 | 0.81 | 0.61 |

## Single encoders（SVR, test）

| Encoder | MAE_V | MAE_A |
|---|---:|---:|
| e5-instruct | 0.488 | 0.788 |
| e5-large | 0.555 | 0.806 |
| DeepSeek-R1 FP16 | 0.517 | 0.799 |
| DeepSeek-Prover FP16 | 0.528 | 0.792 |
| TAIDE FP16 | 0.562 | 0.822 |

樂觀的 `full_dev_on_dev` 數字見 [`notes/report.md`](notes/report.md)。與榜公平比較請以 **test** 為準。

## Data

Shared-task 資料：`data/raw/ROCLING-2025-ST-DSA-MST/`

- Train：合併 `CVAT_1_SD.csv` … `CVAT_5_SD.csv` → **2954** 列（勿用 `CVAT_all_SD.csv`；有壞列）
- Dev（有標籤）：`DSAMST-ValidationSet_ans.csv` → **994**
- Test（有標籤）：`DSAMST-TestSet_ans.csv` → **1541**
- 官方評分：`scoring.py`

原始資料與 embedding `.npy` **不**進 git（見 `.gitignore`）。

## Disclaimer

本倉庫為論文方法的**獨立重現**，**非**原作者官方釋出。

原作者未公開程式碼。此處的實作細節、pipeline 與實驗，皆依論文方法獨立重建，僅供學術研究與可重現性用途。

## Citation & Acknowledgements

若使用本重現或引用原論文方法，請引用：

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
