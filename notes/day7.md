# Day 7 lab notes: e5-large, CatBoost, 8B checks, multi-encoder try

## 1. e5-large (no instruct)

Config: `configs/e5_large.yaml` (`query: ` prefix, no Instruct block).

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| half_dev SVR | 0.595 | 1.067 | 0.713 | 0.442 |
| **test SVR (train+dev)** | **0.555** | **0.806** | **0.739** | **0.534** |
| Paper Table 3 e5-large | 0.554 | 0.810 | 0.726 | 0.531 |

Test is almost the same as the paper’s Table 3 line. Still clearly weaker than e5-instruct (0.488 / 0.788).

## 2. CatBoost

Paper-like settings: iterations=1000, depth=6, lr=0.05.

| Setup | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| half_dev (e5-instruct) | 0.537 | 1.040 | 0.759 | 0.512 |
| test (inside five-model average) | 0.489 | 0.783 | 0.789 | 0.586 |

**Model average with CatBoost on test**

| | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| Day 6 (4 models) | 0.472 | 0.776 | 0.795 | 0.596 |
| **Day 7 (5 models, +CatBoost)** | **0.473** | **0.774** | **0.795** | **0.598** |

Gain is tiny (arousal −0.002). CatBoost is fine as a Table 2 check; not required for the final average.

## 3. Llama3-TAIDE 8B as an embedder?

See [`feasibility_llm_8b.md`](feasibility_llm_8b.md). Short version:

- FP16: **no** (~16GB)
- Official 4-bit / bitsandbytes: **maybe** can load; full embed still risky and not the same as the paper’s precision
- Probe: `python -m src.probe_llm --model taide`

## 4. DeepSeek-R1 4-bit on this PC

- Installed `bitsandbytes` and `accelerate`
- **4-bit load worked**, peak VRAM ~**4.59 GB** (`results/deepseek_4bit_probe.json`)
- **Blocker:** `tokenizer.encode` on **Chinese** returns an empty list (English works). Until that is fixed, I cannot embed the medical texts.
- Recipe in [`feasibility_llm_8b.md`](feasibility_llm_8b.md)

## 5. Multi-encoder average (what 8GB can do)

`src/ensemble_encoders.py` trains one SVR per cached encoder, then averages predictions.

| System | MAE_V | MAE_A | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| e5-instruct SVR | 0.488 | 0.788 | 0.788 | 0.578 |
| e5-large SVR | 0.555 | 0.806 | 0.739 | 0.534 |
| **Two-e5 average** | 0.499 | **0.774** | 0.784 | 0.585 |
| Five-regressor average (instruct) | **0.473** | **0.774** | **0.795** | **0.598** |
| Paper Table 4 Encoders | 0.463 | 0.759 | 0.805 | 0.608 |

**Takeaways**

- Averaging instruct + large: arousal a bit better, valence worse (weak encoder pulls down V).
- Full five-encoder mix (DeepSeek / TAIDE / …): **not realistic on 8GB**.
- Best practical setup here: e5-instruct **model** average (± CatBoost). Encoder averaging only helps when every encoder is strong.

## Commands

```powershell
python -m src.embed --config configs/e5_large.yaml --split all
python -m src.predict_test --config configs/e5_large.yaml --strategy train_full_dev
python -m src.train_boost --model catboost --strategy train_half_dev
python -m src.ensemble_models --strategy train_full_dev --run-official-scoring
python -m src.ensemble_encoders --encoders intfloat/multilingual-e5-large-instruct intfloat/multilingual-e5-large
python -m src.probe_llm --model taide
python -m src.probe_llm --model deepseek --download --load-in-4bit
```
