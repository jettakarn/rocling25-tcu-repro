from __future__ import annotations

"""VRAM / load probe for paper LLMs used as embedding encoders.

Default: estimate only (no download).
  python -m src.probe_llm --model deepseek
  python -m src.probe_llm --model taide
  python -m src.probe_llm --model deepseek --download --load-in-4bit
"""

import argparse
import json
from pathlib import Path

MODEL_PRESETS = {
    "deepseek": {
        "repo": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "fp16_weights_gb": 16.4,
        "bnb4_weights_gb": "4.5–6",
        "notes": "Paper Table 1/3 primary encoder. Reasoning distill of Qwen3-8B.",
    },
    "taide": {
        "repo": "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
        "fp16_weights_gb": 16.0,
        "bnb4_weights_gb": "5–6 (official 4bit repo also exists)",
        "notes": "Paper Table 3 Llama3-TAIDE. Official 4bit: taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit",
        "official_4bit": "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--repo", default=None, help="Override HF repo id.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    preset = MODEL_PRESETS[args.model]
    repo = args.repo or preset["repo"]
    out_path = Path(args.out or f"results/{args.model}_feasibility.json")

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
                "AutoTokenizer + AutoModel (not causal LM generate)",
                "forward → last_hidden_state",
                "mean-pool over non-pad tokens (or last token — pick one and keep fixed)",
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
        from transformers import AutoModel, AutoTokenizer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
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
            text = "病人情況穩定，家屬情緒平靜。"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            inputs = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                add_special_tokens=True,
            )
            if inputs["input_ids"].numel() == 0:
                # Some chat tokenizers need an explicit bos.
                bos = tok.bos_token or tok.eos_token or ""
                inputs = tok(
                    f"{bos}{text}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    add_special_tokens=True,
                )
            report["n_tokens"] = int(inputs["input_ids"].numel())
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
                hidden = out.last_hidden_state
                mask = inputs.get("attention_mask")
                if mask is not None:
                    mask = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                else:
                    pooled = hidden.mean(dim=1)
            report["forward_ok"] = True
            report["probe_hidden_shape"] = list(pooled.shape)
            report["peak_vram_after_forward_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
            del model, tok, out, pooled, hidden
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            report["load_ok"] = report.get("load_ok", False)
            report["forward_ok"] = False
            report["load_error"] = repr(e)
            report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
