#!/usr/bin/env bash
set -euo pipefail
source /workspace/venv/bin/activate
cd /workspace/rocling-dsa-repro

python <<'PY'
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
repo = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
cands = []
cands.append(("auto", lambda: AutoTokenizer.from_pretrained(repo, trust_remote_code=True)))
cands.append(("auto_slow", lambda: AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)))
try:
    cands.append(("llama_fast", lambda: LlamaTokenizerFast.from_pretrained(repo, trust_remote_code=True)))
except Exception:
    pass
try:
    cands.append(("llama_slow", lambda: LlamaTokenizer.from_pretrained(repo, trust_remote_code=True)))
except Exception:
    pass

best = None
for name, fn in cands:
    try:
        tok = fn()
        zh = tok.encode("病人情況穩定。", add_special_tokens=True)
        en = tok.encode("Hello world", add_special_tokens=True)
        print(f"{name}: cls={type(tok).__name__} zh_n={len(zh)} zh={zh[:16]} en_n={len(en)} en={en[:16]}")
        if len(zh) > 0 and best is None:
            best = (name, tok)
    except Exception as e:
        print(f"{name}: ERR {type(e).__name__}: {e}")

if best is None:
    raise SystemExit("No tokenizer produced non-empty Chinese ids")
print("BEST", best[0])
PY
