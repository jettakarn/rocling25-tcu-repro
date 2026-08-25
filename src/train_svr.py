from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import SVR

from src.data_loader import load_table
from src.embed import encoder_slug
from src.metrics import evaluate_va, format_metrics


def load_xy(csv_path: str, npy_path: Path):
    df = load_table(csv_path)
    x = np.load(npy_path)
    if len(df) != len(x):
        raise ValueError(f"row mismatch: {csv_path} n={len(df)} vs {npy_path} n={len(x)}")
    return df, x


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def train_one(x, y, C: float, epsilon: float) -> SVR:
    model = SVR(kernel="rbf", C=C, epsilon=epsilon)
    model.fit(x, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--strategy",
        choices=["train", "train_half_dev", "train_full_dev"],
        default="train_half_dev",
        help="Paper Table 1 training combinations.",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="L2-normalize cached embeddings before SVR (Day 3 probe).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")
    if args.l2_normalize:
        train_x = l2_normalize(train_x)
        dev_x = l2_normalize(dev_x)
        print("l2-normalize: on")

    if args.strategy == "train":
        x = train_x
        yv = train_df["valence"].to_numpy()
        ya = train_df["arousal"].to_numpy()
        eval_mask = np.ones(len(dev_df), dtype=bool)
    elif args.strategy == "train_full_dev":
        x = np.vstack([train_x, dev_x])
        yv = np.concatenate([train_df["valence"], dev_df["valence"]])
        ya = np.concatenate([train_df["arousal"], dev_df["arousal"]])
        eval_mask = np.ones(len(dev_df), dtype=bool)
        print("warning: train_full_dev evaluates on the same labeled dev set (optimistic).")
    else:
        idx = np.arange(len(dev_df))
        rng.shuffle(idx)
        half = len(idx) // 2
        add, hold = idx[:half], idx[half:]
        x = np.vstack([train_x, dev_x[add]])
        yv = np.concatenate([train_df["valence"].to_numpy(), dev_df.iloc[add]["valence"].to_numpy()])
        ya = np.concatenate([train_df["arousal"].to_numpy(), dev_df.iloc[add]["arousal"].to_numpy()])
        eval_mask = np.zeros(len(dev_df), dtype=bool)
        eval_mask[hold] = True

    svr = cfg["svr"]
    model_v = train_one(x, yv, svr["C"], svr["epsilon"])
    model_a = train_one(x, ya, svr["C"], svr["epsilon"])

    pred_v = model_v.predict(dev_x)
    pred_a = model_a.predict(dev_x)
    metrics = evaluate_va(
        dev_df.loc[eval_mask, "valence"],
        pred_v[eval_mask],
        dev_df.loc[eval_mask, "arousal"],
        pred_a[eval_mask],
    )
    print(args.strategy, format_metrics(metrics))

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_l2" if args.l2_normalize else ""
    out = out_dir / f"{enc}_{args.strategy}{suffix}.json"
    payload = {
        "encoder": cfg["encoder_name"],
        "strategy": args.strategy,
        "l2_normalize": bool(args.l2_normalize),
        "n_train": int(len(x)),
        "n_eval": int(eval_mask.sum()),
        "metrics": metrics,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
