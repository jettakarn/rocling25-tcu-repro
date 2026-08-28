from __future__ import annotations

"""VRAM / load probe for paper LLMs used as embedding encoders.

Default: estimate only (no download).
  python -m src.probe_llm --model deepseek
  python -m src.probe_llm --model taide
  python -m src.probe_llm --model deepseek --download --load-in-4bit
  python -m src.probe_llm --model deepseek --download --load-in-4bit --smoke-n 50
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_PRESETS = {
    "deepseek": {
        "repo": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "fp16_weights_gb": 16.4,
        "bnb4_weights_gb": "4.5–6",
        "notes": "Paper Table 1/3 primary encoder. Reasoning distill of Qwen3-8B.",
        "force_qwen2_tokenizer": True,
    },
    "prover": {
        "repo": "deepseek-ai/DeepSeek-Prover-V1.5-RL",
        "fp16_weights_gb": 14.0,
        "bnb4_weights_gb": "4–5",
        "notes": "Paper Table 3 DeepSeek-Prover-V1.5-RL (7B). Mean-pool last_hidden_state.",
        "force_qwen2_tokenizer": False,
    },
    "taide": {
        "repo": "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
        "fp16_weights_gb": 16.0,
        "bnb4_weights_gb": "5–6 (official 4bit repo also exists)",
        "notes": "Paper Table 3 Llama3-TAIDE. Official 4bit: taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit",
        "official_4bit": "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit",
        "force_qwen2_tokenizer": False,
    },
}

PROBE_ZH = "病人情況穩定，家屬情緒平靜。"
PROBE_EN = "Hello world"


def load_llm_tokenizer(repo: str, *, force_qwen2: bool = False):
    """Load tokenizer; DeepSeek-R1-Qwen3 needs Qwen2 BPE, not LlamaTokenizerFast.

    The HF repo sets tokenizer_class=LlamaTokenizerFast. Under transformers 5.x that
    path yields empty input_ids for Chinese (English still tokenizes, but with
    different ids than tokenizer.json). Force tokenizer_type='qwen2' / Qwen2Tokenizer.
    """
    from transformers import AutoTokenizer

    if force_qwen2:
        try:
            tok = AutoTokenizer.from_pretrained(
                repo, trust_remote_code=True, tokenizer_type="qwen2"
            )
        except TypeError:
            from transformers import Qwen2TokenizerFast

            tok = Qwen2TokenizerFast.from_pretrained(repo, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def diagnose_tokenizer(tok, force_qwen2: bool) -> dict:
    zh_ids = tok.encode(PROBE_ZH, add_special_tokens=True)
    en_ids = tok.encode(PROBE_EN, add_special_tokens=True)
    zh_call = tok(
        PROBE_ZH,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    return {
        "tokenizer_class": type(tok).__name__,
        "is_fast": bool(getattr(tok, "is_fast", False)),
        "force_qwen2": force_qwen2,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "zh_n_tokens": len(zh_ids),
        "zh_input_ids_head": zh_ids[:16],
        "en_n_tokens": len(en_ids),
        "en_input_ids_head": en_ids[:16],
        "zh_call_numel": int(zh_call["input_ids"].numel()),
        "chinese_ok": len(zh_ids) > 0 and int(zh_call["input_ids"].numel()) > 0,
    }


def mean_pool(hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def smoke_embed(model, tok, texts: list[str], max_length: int = 512) -> dict:
    import torch

    device = next(model.parameters()).device
    vecs = []
    empty = 0
    peak = 0.0
    with torch.no_grad():
        for text in texts:
            inputs = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            n = int(inputs["input_ids"].numel())
            if n == 0:
                empty += 1
                vecs.append(None)
                continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, use_cache=False)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            vecs.append(pooled.squeeze(0).float().cpu().numpy())
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
    ok = [v for v in vecs if v is not None]
    stacked = np.stack(ok, axis=0) if ok else np.zeros((0, 0), dtype=np.float32)
    return {
        "n_texts": len(texts),
        "n_embedded": len(ok),
        "n_empty_tokenize": empty,
        "embed_shape": list(stacked.shape),
        "embed_norm_mean": float(np.linalg.norm(stacked, axis=1).mean()) if len(ok) else None,
        "peak_vram_gb": round(peak, 2),
        "ok": empty == 0 and len(ok) == len(texts),
    }


def _sample_smoke_texts(n: int, seed: int = 42) -> list[str]:
    from src.data_loader import load_table

    train = load_table("data/processed/train.csv")
    dev = load_table("data/processed/dev.csv")
    rng = np.random.default_rng(seed)
    n_train = min(n // 2 + n % 2, len(train))
    n_dev = min(n - n_train, len(dev))
    ti = rng.choice(len(train), size=n_train, replace=False)
    di = rng.choice(len(dev), size=n_dev, replace=False)
    texts = train.iloc[ti]["text"].tolist() + dev.iloc[di]["text"].tolist()
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--repo", default=None, help="Override HF repo id.")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=0,
        help="If >0 with --download, mean-pool embed this many train+dev lines.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--no-force-qwen2-tokenizer",
        action="store_true",
        help="Use AutoTokenizer default (reproduces Chinese-empty bug on DeepSeek).",
    )
    args = parser.parse_args()

    import torch

    preset = MODEL_PRESETS[args.model]
    repo = args.repo or preset["repo"]
    out_path = Path(args.out or f"results/{args.model}_feasibility.json")
    force_qwen2 = bool(preset.get("force_qwen2_tokenizer")) and not args.no_force_qwen2_tokenizer

    report: dict = {
        "preset": args.model,
        "repo": repo,
        "gpu": None,
        "vram_total_gb": None,
        "estimates": {
            "fp16_weights_gb": preset["fp16_weights_gb"],
            "bnb_4bit_weights_gb": preset["bnb4_weights_gb"],
        },
        "preset_notes": preset["notes"],
        "official_4bit": preset.get("official_4bit"),
        "quant_implementation": {
            "pip": "pip install bitsandbytes accelerate",
            "config": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            },
            "embed_recipe": [
                "AutoTokenizer with tokenizer_type='qwen2' for DeepSeek-R1-Qwen3 (not LlamaTokenizerFast)",
                "AutoModel (not causal LM generate)",
                "forward → last_hidden_state",
                "mean-pool over non-pad tokens",
                "batch_size=1, max_length≤512 on 8GB",
                "cache .npy then reuse existing SVR / ensemble scripts",
            ],
            "risks": [
                "Quant ≠ paper FP precision; numbers not claimable as Table 3 reproduction",
                "Long medical texts inflate activations → OOM",
                "Full corpus embed (~5.5k texts) is hours + fragile",
                "Windows bitsandbytes historically flaky; may need WSL2",
            ],
        },
        "verdict": None,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["vram_total_gb"] = round(props.total_memory / 1024**3, 2)

    import importlib.util as u

    report["bitsandbytes"] = u.find_spec("bitsandbytes") is not None
    report["accelerate"] = u.find_spec("accelerate") is not None

    vram = report["vram_total_gb"] or 0
    report["verdict"] = {
        "fp16_embedding": "no" if vram and vram <= 8.5 else "maybe",
        "bnb_4bit_embedding": "maybe — install bnb; probe single forward first",
        "full_paper_table3_reproduction": "no on 8GB without quant deviation",
        "multi_encoder_with_8b": "no for five 8B encoders; yes for dual e5 (+optional 4bit one-shot later)",
        "recommendation": (
            "Keep e5-instruct + e5-large encoder ensemble on 8GB. "
            "Treat DeepSeek/TAIDE 4-bit as optional research probe, not Week-1 critical path."
        ),
    }

    if args.download:
        if not torch.cuda.is_available():
            raise SystemExit("Need CUDA")
        from transformers import AutoModel

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            tok = load_llm_tokenizer(repo, force_qwen2=force_qwen2)
            report["tokenizer"] = diagnose_tokenizer(tok, force_qwen2)
            if not report["tokenizer"]["chinese_ok"]:
                raise RuntimeError(
                    "Chinese tokenization still empty; refuse forward. "
                    "For DeepSeek use force_qwen2 (default) / tokenizer_type='qwen2'."
                )

            if args.load_in_4bit:
                if not report["bitsandbytes"] or not report["accelerate"]:
                    raise SystemExit("pip install bitsandbytes accelerate")
                from transformers import BitsAndBytesConfig

                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModel.from_pretrained(
                    repo,
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModel.from_pretrained(
                    repo,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            report["load_ok"] = True
            report["peak_vram_after_load_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )

            inputs = tok(
                PROBE_ZH,
                return_tensors="pt",
                truncation=True,
                max_length=min(128, args.max_length),
                add_special_tokens=True,
            )
            report["n_tokens"] = int(inputs["input_ids"].numel())
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
                pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            report["forward_ok"] = True
            report["probe_hidden_shape"] = list(pooled.shape)
            report["peak_vram_after_forward_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )

            if args.smoke_n > 0:
                texts = _sample_smoke_texts(args.smoke_n)
                report["smoke"] = smoke_embed(
                    model, tok, texts, max_length=args.max_length
                )
                report["verdict"]["bnb_4bit_embedding"] = (
                    "ok on smoke" if report["smoke"]["ok"] else "smoke failed"
                )

            del model, tok, out, pooled
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            report["load_ok"] = report.get("load_ok", False)
            report["forward_ok"] = False
            report["load_error"] = repr(e)
            if torch.cuda.is_available():
                report["peak_vram_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**3, 2
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
