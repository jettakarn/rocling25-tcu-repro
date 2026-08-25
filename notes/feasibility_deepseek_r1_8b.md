# DeepSeek-R1-0528-Qwen3-8B: can it run on RTX 3070 8GB?

Paper’s main encoder name: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (~8.2B params).  
Old probe script: `python -m src.probe_deepseek` (no download by default).  
Newer checks: `python -m src.probe_llm --model deepseek` and [`feasibility_llm_8b.md`](feasibility_llm_8b.md).

## Machine (first check)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| VRAM | **8.0 GB** |
| bitsandbytes / accelerate | later installed for the 4-bit try |

## Rough memory sizes (public estimates)

| Load style | Weights | Extra cost | Fits 8GB? |
|---|---:|---|---|
| FP16 / BF16 (transformers) | ~16.4 GB | ~18+ GB | **no** |
| Q8 GGUF | ~8–9 GB | often over 8GB | **risky / no** |
| Q4_K_M GGUF (chat) | ~4.7–5.5 GB | ~6–7.6 GB (short context) | **chat: maybe** |
| bitsandbytes NF4 | ~4.5–6 GB | activations; long text may OOM | **embed: maybe** |

## How the paper uses the model vs chat tools

The paper uses the LLM as an **embedding model** (hidden states → SVR).  
That is **not** the same as asking Ollama to output a score in chat.

| Path | Matches paper? | On 8GB |
|---|---|---|
| `AutoModel` FP16 + mean pool | yes | **no** (weights too big) |
| `AutoModel` 4-bit + mean pool, batch=1 | close, but quantized | maybe; long texts may OOM; full corpus is slow |
| GGUF / Ollama chat | **no** | chat may work; hard to feed SVR |
| Other embed servers | depends | needs new code, not `embed.py` |

## Early conclusion (Day 6)

1. Do not make DeepSeek the main path this week — FP16 does not fit.
2. If you only want a smoke test: install `bitsandbytes` + `accelerate`, then run a 4-bit load + one forward.
3. Quantized scores are **not** a clean copy of paper Tables 1/3.
4. Averaging e5 regressors already got close to paper Table 4 “Models”.

## Later update (Day 7)

4-bit load **did** work (~4.6GB), but Chinese tokenization returned empty strings, so full medical embedding is blocked for now. Details in [`feasibility_llm_8b.md`](feasibility_llm_8b.md).
