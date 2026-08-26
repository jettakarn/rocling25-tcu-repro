from __future__ import annotations

"""4-bit DeepSeek mean-pool embeddings (subset / exploratory).

  python -m src.embed_llm --n-train 300 --n-dev 200 --compare-e5

Writes under data/embeddings/deepseek_r1_8b_nf4/.
Results are labeled quantized=true — not paper Table 1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.svm import SVR
from tqdm import tqdm

from src.data_loader import load_table
from src.metrics import evaluate_va, format_metrics
from src.probe_llm import load_llm_tokenizer, mean_pool

DEFAULT_REPO = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OUT_SLUG = "deepseek_r1_8b_nf4"
E5_SLUG = "intfloat__multilingual-e5-large-instruct"


def embed_texts_llm(
    model,
    tok,
    texts: list[str],
    *,
    max_length: int = 512,
) -> np.ndarray:
    device = next(model.parameters()).device
    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for text in tqdm(texts, desc="embed_llm", leave=True):
            inputs = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            if int(inputs["input_ids"].numel()) == 0:
                raise RuntimeError(f"empty tokenization for text={text[:80]!r}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, use_cache=False)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            vecs.append(pooled.squeeze(0).float().cpu().numpy())
    return np.stack(vecs, axis=0)


def load_nf4_model(repo: str):
    from transformers import AutoModel, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModel.from_pretrained(
        repo,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )


def sample_indices(n_total: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    n_take = min(n_take, n_total)
    return np.sort(rng.choice(n_total, size=n_take, replace=False))


def half_dev_svr(
    train_x: np.ndarray,
    train_y_v: np.ndarray,
    train_y_a: np.ndarray,
    dev_x: np.ndarray,
    dev_y_v: np.ndarray,
    dev_y_a: np.ndarray,
    *,
    seed: int,
    C: float,
    epsilon: float,
) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dev_x))
    rng.shuffle(idx)
    half = len(idx) // 2
    add, hold = idx[:half], idx[half:]
    x = np.vstack([train_x, dev_x[add]])
    yv = np.concatenate([train_y_v, dev_y_v[add]])
    ya = np.concatenate([train_y_a, dev_y_a[add]])
    mv = SVR(kernel="rbf", C=C, epsilon=epsilon).fit(x, yv)
    ma = SVR(kernel="rbf", C=C, epsilon=epsilon).fit(x, ya)
    pred_v = mv.predict(dev_x)
    pred_a = ma.predict(dev_x)
    metrics = evaluate_va(dev_y_v[hold], pred_v[hold], dev_y_a[hold], pred_a[hold])
    return {
        "strategy": "train_half_dev",
        "n_train": int(len(x)),
        "n_eval": int(len(hold)),
        "hold_indices_within_subset_dev": hold.tolist(),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-dev", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--compare-e5",
        action="store_true",
        help="SVR half_dev on DeepSeek subset vs e5-instruct same rows.",
    )
    parser.add_argument("--skip-embed", action="store_true", help="Reuse existing .npy")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Need CUDA")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    rng = np.random.default_rng(seed)
    out_dir = Path(args.out_dir or Path(cfg["data"]["embeddings_dir"]) / OUT_SLUG)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_table(cfg["data"]["train_path"])
    dev_df = load_table(cfg["data"]["dev_path"])
    train_idx = sample_indices(len(train_df), args.n_train, rng)
    # fresh rng stream for dev so changing n_train alone doesn't reshuffle dev
    rng_dev = np.random.default_rng(seed + 1)
    dev_idx = sample_indices(len(dev_df), args.n_dev, rng_dev)

    train_sub = train_df.iloc[train_idx].reset_index(drop=True)
    dev_sub = dev_df.iloc[dev_idx].reset_index(drop=True)

    meta = {
        "repo": args.repo,
        "quantized": True,
        "quant": "nf4_double_quant_fp16_compute",
        "pooling": "mean_nonpad",
        "max_length": args.max_length,
        "batch_size": 1,
        "seed": seed,
        "n_train": int(len(train_sub)),
        "n_dev": int(len(dev_sub)),
        "train_indices": train_idx.tolist(),
        "dev_indices": dev_idx.tolist(),
        "claim": "exploratory 4-bit subset — NOT paper Table 1 reproduction",
    }

    train_npy = out_dir / "train.npy"
    dev_npy = out_dir / "dev.npy"

    if args.skip_embed and train_npy.exists() and dev_npy.exists():
        train_x = np.load(train_npy)
        dev_x = np.load(dev_npy)
        print(f"reused {train_npy} {train_x.shape}, {dev_npy} {dev_x.shape}")
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tok = load_llm_tokenizer(args.repo, force_qwen2=True)
        # sanity: Chinese must be non-empty
        if len(tok.encode("病人情況穩定。")) == 0:
            raise SystemExit("Chinese tokenization empty; abort embed")
        model = load_nf4_model(args.repo)
        meta["peak_vram_after_load_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 2
        )
        train_x = embed_texts_llm(
            model, tok, train_sub["text"].tolist(), max_length=args.max_length
        )
        dev_x = embed_texts_llm(
            model, tok, dev_sub["text"].tolist(), max_length=args.max_length
        )
        meta["peak_vram_after_embed_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 2
        )
        np.save(train_npy, train_x)
        np.save(dev_npy, dev_x)
        print(f"saved {train_npy} {train_x.shape}")
        print(f"saved {dev_npy} {dev_x.shape}")
        del model, tok
        torch.cuda.empty_cache()

    (out_dir / "subset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    svr_cfg = cfg["svr"]
    ds_result = half_dev_svr(
        train_x,
        train_sub["valence"].to_numpy(),
        train_sub["arousal"].to_numpy(),
        dev_x,
        dev_sub["valence"].to_numpy(),
        dev_sub["arousal"].to_numpy(),
        seed=seed,
        C=float(svr_cfg["C"]),
        epsilon=float(svr_cfg["epsilon"]),
    )
    ds_result.update(
        {
            "encoder": args.repo,
            "quantized": True,
            "subset": True,
            "n_train_subset": int(len(train_sub)),
            "n_dev_subset": int(len(dev_sub)),
            "claim": "NOT paper Table 1",
        }
    )
    print("deepseek_nf4", format_metrics(ds_result["metrics"]))

    payload: dict = {"deepseek_r1_8b_nf4": ds_result, "meta": meta}

    if args.compare_e5:
        e5_dir = Path(cfg["data"]["embeddings_dir"]) / E5_SLUG
        e5_train = np.load(e5_dir / "train.npy")[train_idx]
        e5_dev = np.load(e5_dir / "dev.npy")[dev_idx]
        e5_result = half_dev_svr(
            e5_train,
            train_sub["valence"].to_numpy(),
            train_sub["arousal"].to_numpy(),
            e5_dev,
            dev_sub["valence"].to_numpy(),
            dev_sub["arousal"].to_numpy(),
            seed=seed,
            C=float(svr_cfg["C"]),
            epsilon=float(svr_cfg["epsilon"]),
        )
        e5_result.update(
            {
                "encoder": "intfloat/multilingual-e5-large-instruct",
                "quantized": False,
                "subset": True,
                "same_rows_as": OUT_SLUG,
                "n_train_subset": int(len(train_sub)),
                "n_dev_subset": int(len(dev_sub)),
            }
        )
        print("e5_instruct_same_subset", format_metrics(e5_result["metrics"]))
        payload["e5_instruct_same_subset"] = e5_result
        payload["delta_deepseek_minus_e5"] = {
            k: ds_result["metrics"][k] - e5_result["metrics"][k]
            for k in ("mae_v", "mae_a", "pcc_v", "pcc_a")
        }

    results_dir = Path(cfg["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{OUT_SLUG}_subset_half_dev.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
