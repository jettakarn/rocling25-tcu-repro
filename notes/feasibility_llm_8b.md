# 8B models as embedders on a 3070 8GB card

Models from the paper:

- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `taide/Llama3-TAIDE-LX-8B-Chat-Alpha1` (also has an official `…-4bit` repo)

Script: `python -m src.probe_llm --model {deepseek|taide}`

## Shared conclusions

| Path | Feasible? | Note |
|---|---|---|
| FP16 / BF16 `AutoModel` embed | **no** | weights alone ~16GB |
| bitsandbytes NF4 embed | **maybe** | ~5–6GB weights; long text may OOM |
| GGUF / Ollama chat | chat maybe | **not** hidden-state → SVR |
| Exact paper Table 1/3 numbers | **no** if quantized | only exploratory |

## Llama3-TAIDE

- Tuned for Traditional Chinese office / chat tasks; appears in paper Table 3.
- Official 4-bit pack: `taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit` (they warn quality may drop).
- If trying embeddings: start with official 4-bit or bnb NF4, batch=1, max_length≤512, try ~50 sentences first.

## DeepSeek 4-bit recipe

```powershell
pip install bitsandbytes accelerate
python -m src.probe_llm --model deepseek --download --load-in-4bit
```

### What happened on this PC (2026-08-25)

| Check | Result |
|---|---|
| 4-bit NF4 load | **ok**, peak ~**4.59 GB** |
| One forward | works in principle (hidden size 4096) |
| Chinese `tokenizer.encode` | **empty list** (English is fine) |
| Full medical embedding | **blocked** by tokenization |

So: the GPU can hold a 4-bit DeepSeek, but I still need a working Chinese tokenizer before writing a full `embed_llm` script.

Example config:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
# then AutoModel + mean-pool over last_hidden_state
```

## Can I do multi-encoder fusion?

| Idea | On 8GB? | In this project |
|---|---|---|
| Paper’s five-encoder average | needs several 8B models | **no** |
| e5-instruct + e5-large SVR average | **yes** | ran; arousal a bit better, valence worse |
| e5-instruct model average | **yes** | best practical choice (± CatBoost) |
| e5 + one 4-bit 8B | maybe later | only after tokenizer / load is stable |

**Bottom line:** on this machine, “multi-encoder” tops out at **two e5 models**. The full paper encoder mix needs more VRAM. Quantized 8B is optional research, not required to finish the report.
