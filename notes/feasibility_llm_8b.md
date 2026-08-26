# 8B models as embedders on a 3070 8GB card

Models from the paper:

- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `taide/Llama3-TAIDE-LX-8B-Chat-Alpha1` (also has an official `…-4bit` repo)

Script: `python -m src.probe_llm --model {deepseek|taide}`

## Shared conclusions

| Path | Feasible? | Note |
|---|---|---|
| FP16 / BF16 `AutoModel` embed | **no** | weights alone ~16GB |
| bitsandbytes NF4 embed | **maybe → yes (subset)** | ~4.6GB weights; Chinese tokenizer fixed |
| GGUF / Ollama chat | chat maybe | **not** hidden-state → SVR |
| Exact paper Table 1/3 numbers | **no** if quantized | only exploratory |

## Llama3-TAIDE

- Tuned for Traditional Chinese office / chat tasks; appears in paper Table 3.
- Official 4-bit pack: `taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit` (they warn quality may drop).
- If trying embeddings: start with official 4-bit or bnb NF4, batch=1, max_length≤512, try ~50 sentences first.

## DeepSeek 4-bit recipe

```powershell
pip install bitsandbytes accelerate
python -m src.probe_llm --model deepseek --download --load-in-4bit --smoke-n 50
python -m src.embed_llm --subset-n 400 --compare-e5
```

### Tokenizer bug (fixed 2026-08-26)

| Check | Before | After |
|---|---|---|
| `AutoTokenizer` default | class **`LlamaTokenizer` / Fast** (from `tokenizer_class` in repo) | force **`Qwen2Tokenizer`** via `tokenizer_type="qwen2"` |
| Chinese `encode("病人…")` | **`[]`** | **10 ids** (matches raw `tokenizer.json`) |
| English `encode("Hello world")` | `[39, 95292]` (wrong vs json) | `[9707, 1879]` (matches json) |

Root cause: HF config advertises `LlamaTokenizerFast`, but the vocab is Qwen BPE (`tokenizer.json` only). The Llama wrapper tokenizes English with wrong ids and drops CJK to empty. Fix in `src.probe_llm.load_llm_tokenizer(..., force_qwen2=True)` / `src.embed_llm`.

### What happened on this PC

| Check | 2026-08-25 | 2026-08-26 (A1) |
|---|---|---|
| 4-bit NF4 load | **ok**, peak ~**4.59 GB** | same **4.59 GB** |
| One forward | blocked by empty Chinese ids | **ok**, hidden **(1, 4096)** |
| Chinese tokenize | empty | **ok** (`chinese_ok: true`) |
| Smoke mean-pool 50 train+dev | — | **50/50**, shape `(50, 4096)`, peak **4.63 GB** |
| Full medical embedding | blocked | unblocked for subset (A2) |

Example config:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
# AutoTokenizer(..., tokenizer_type="qwen2")  # required for DeepSeek-R1-Qwen3
# then AutoModel + mean-pool over last_hidden_state
```

Artifact: `results/deepseek_feasibility_a1.json`.

### A2 subset SVR (2026-08-26) — `quantized=true`

- Embed: `src/embed_llm.py` → `data/embeddings/deepseek_r1_8b_nf4/` (300 train + 200 dev, batch=1, max_length=512, NF4).
- Protocol: same `train_half_dev` split on that subset; e5-instruct compared on **identical rows**.
- **Not** paper Table 1.

| Encoder | MAE_V | MAE_A | PCC_V | PCC_A | n_train / n_eval |
|---|---|---|---|---|---|
| DeepSeek NF4 (subset) | 0.701 | 1.088 | 0.566 | 0.372 | 400 / 100 |
| e5-instruct (same subset) | **0.558** | **1.077** | **0.737** | **0.434** | 400 / 100 |

e5 wins on this small holdout. 4-bit DeepSeek is runnable but weaker here; full FP corpus needs Phase B (≥16GB).

Artifact: `results/deepseek_r1_8b_nf4_subset_half_dev.json`.

## Can I do multi-encoder fusion?

| Idea | On 8GB? | In this project |
|---|---|---|
| Paper’s five-encoder average | needs several 8B models | **no** |
| e5-instruct + e5-large SVR average | **yes** | ran; arousal a bit better, valence worse |
| e5-instruct model average | **yes** | best practical choice (± CatBoost) |
| e5 + one 4-bit 8B | subset only | after tokenizer fix (A2) |

**Bottom line:** on this machine, full paper encoder mix still needs more VRAM. Quantized DeepSeek is now **runnable for subset probes** (`quantized=true`); do **not** claim paper Table 1.
