from __future__ import annotations

"""FP16/BF16 full-corpus LLM mean-pool embeddings (paper Table 3 path).

Needs ≥16GB VRAM for 8B models. Example on RunPod:

  python -m src.embed_llm_full --model deepseek --split all
  python -m src.embed_llm_full --model prover --split all

Writes data/embeddings/<encoder_slug>/{train,dev,test}.npy + embed_meta.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data_loader import load_table
from src.embed import encoder_slug
from src.probe_llm import MODEL_PRESETS, load_llm_tokenizer, mean_pool


def load_fp_model(repo: str, dtype: torch.dtype):
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        repo,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )


def embed_texts_llm(
    model,
    tok,
    texts: list[str],
    *,
    max_length: int = 512,
    batch_size: int = 1,
) -> np.ndarray:
    device = next(model.parameters()).device
    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="embed_llm_full"):
            chunk = texts[start : start + batch_size]
            inputs = tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                add_special_tokens=True,
            )
            if int(inputs["input_ids"].numel()) == 0:
                raise RuntimeError(f"empty tokenization near index={start}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, use_cache=False)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            vecs.append(pooled.float().cpu().numpy())
    return np.concatenate(vecs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-corpus FP LLM embeddings.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS),
        default=None,
        help="Preset from probe_llm.MODEL_PRESETS.",
    )
    parser.add_argument("--repo", default=None, help="Override HF repo id.")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "both", "all"],
        default="all",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--force-qwen2-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override preset tokenizer forcing (DeepSeek-R1 needs Qwen2).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Need CUDA for FP LLM embed")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model:
        preset = MODEL_PRESETS[args.model]
        repo = args.repo or preset["repo"]
        force_qwen2 = (
            bool(preset.get("force_qwen2_tokenizer"))
            if args.force_qwen2_tokenizer is None
            else bool(args.force_qwen2_tokenizer)
        )
    elif args.repo:
        repo = args.repo
        force_qwen2 = bool(args.force_qwen2_tokenizer)
    else:
        raise SystemExit("Pass --model or --repo")

    max_length = int(args.max_length if args.max_length is not None else cfg.get("max_length", 512))
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    out_dir = Path(cfg["data"]["embeddings_dir"]) / encoder_slug(repo)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: list[tuple[str, str]] = []
    if args.split in {"train", "both", "all"}:
        splits.append(("train", cfg["data"]["train_path"]))
    if args.split in {"dev", "both", "all"}:
        splits.append(("dev", cfg["data"]["dev_path"]))
    if args.split in {"test", "all"}:
        splits.append(("test", cfg["data"].get("test_path", "data/processed/test.csv")))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = load_llm_tokenizer(repo, force_qwen2=force_qwen2)
    if len(tok.encode("病人情況穩定。")) == 0:
        raise SystemExit("Chinese tokenization empty; abort (check tokenizer)")

    model = load_fp_model(repo, dtype)
    peak_load = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    meta = {
        "repo": repo,
        "quantized": False,
        "dtype": args.dtype,
        "pooling": "mean_nonpad",
        "max_length": max_length,
        "batch_size": args.batch_size,
        "force_qwen2_tokenizer": force_qwen2,
        "peak_vram_after_load_gb": peak_load,
        "claim": "FP full-corpus — paper Table 3 encoder path",
        "splits": {},
    }

    for name, path in splits:
        df = load_table(path)
        texts = df["text"].tolist()
        vecs = embed_texts_llm(
            model,
            tok,
            texts,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        out = out_dir / f"{name}.npy"
        np.save(out, vecs)
        meta["splits"][name] = {"n": int(len(df)), "shape": list(vecs.shape), "path": str(out)}
        print(f"saved {out} shape={vecs.shape}")

    meta["peak_vram_after_embed_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (out_dir / "embed_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'embed_meta.json'}")
    print(f"peak_vram_gb={meta['peak_vram_after_embed_gb']}")


if __name__ == "__main__":
    main()
