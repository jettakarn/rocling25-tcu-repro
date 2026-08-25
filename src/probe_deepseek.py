from __future__ import annotations

"""Probe whether DeepSeek-R1-0528-Qwen3-8B can load on this GPU for embedding.

Does NOT download by default. Pass --download to attempt a real load.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="HF repo id used in the TCU paper.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download/load the model (multi-GB). Default: estimate only.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Try bitsandbytes 4-bit load (requires bitsandbytes + accelerate).",
    )
    parser.add_argument(
        "--out",
        default="results/deepseek_r1_8b_feasibility.json",
    )
    args = parser.parse_args()

    import torch

    report: dict = {
        "model": args.model,
        "gpu": None,
        "vram_total_gb": None,
        "estimates": {
            "fp16_weights_gb": 16.4,
            "fp16_runtime_gb": "~18+",
            "bnb_4bit_weights_gb": "~4.5–6",
            "gguf_q4_k_m_runtime_gb": "~5.5–7.6 (chat); embedding path differs",
        },
        "verdict_for_embedding_on_8gb": None,
        "notes": [],
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["vram_total_gb"] = round(props.total_memory / 1024**3, 2)
    else:
        report["notes"].append("CUDA unavailable; cannot probe GPU load.")

    try:
        import importlib.util as u

        report["bitsandbytes"] = u.find_spec("bitsandbytes") is not None
        report["accelerate"] = u.find_spec("accelerate") is not None
    except Exception as e:  # noqa: BLE001
        report["notes"].append(f"package probe failed: {e}")

    vram = report["vram_total_gb"] or 0
    if vram and vram <= 8.5:
        report["verdict_for_embedding_on_8gb"] = {
            "fp16_transformers_forward": "no — weights alone ~16GB",
            "bnb_nf4_hidden_mean_pool": "maybe — need bitsandbytes; batch=1, max_length≤512; expect OOM risk on long texts",
            "gguf_chat_via_ollama": "yes for generation — NOT a drop-in embedding encoder for paper SVR pipeline",
            "recommendation": "Do not put DeepSeek on Day 6–7 critical path. Prefer e5 + model ensemble. If probing, install bitsandbytes and try --download --load-in-4bit with batch_size=1.",
        }
    else:
        report["verdict_for_embedding_on_8gb"] = {
            "recommendation": "VRAM > 8GB; FP16 or 4-bit embedding extraction is more realistic.",
        }

    if args.download:
        if not torch.cuda.is_available():
            raise SystemExit("Need CUDA for download probe")
        from transformers import AutoModel, AutoTokenizer

        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated() / 1024**3
        try:
            if args.load_in_4bit:
                if not report.get("bitsandbytes") or not report.get("accelerate"):
                    raise SystemExit("Install bitsandbytes and accelerate for --load-in-4bit")
                from transformers import BitsAndBytesConfig

                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    args.model,
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Will almost certainly OOM on 8GB
                tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    args.model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            after = torch.cuda.max_memory_allocated() / 1024**3
            report["load_ok"] = True
            report["peak_vram_gb"] = round(after, 2)
            report["allocated_delta_gb"] = round(after - before, 2)
            # Tiny forward to estimate activation cost
            text = "病人情況穩定，家屬情緒平靜。"
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                hidden = out.last_hidden_state
                pooled = hidden.mean(dim=1)
            report["probe_hidden_shape"] = list(pooled.shape)
            report["peak_vram_after_forward_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
            del model, tok, out, hidden, pooled
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            report["load_ok"] = False
            report["load_error"] = repr(e)
            report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
