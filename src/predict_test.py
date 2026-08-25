from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.embed import encoder_slug
from src.metrics import evaluate_va, format_metrics
from src.train_svr import load_xy, train_one


def clip_va(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 9.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SVR on train(+dev) and predict labeled test set."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--strategy",
        choices=["train", "train_full_dev"],
        default="train_full_dev",
        help="train_full_dev aligns with paper Table 3 encoder comparison.",
    )
    parser.add_argument(
        "--run-official-scoring",
        action="store_true",
        help="Also run shared-task scoring.py against DSAMST-TestSet_ans.csv.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc
    test_csv = cfg["data"].get("test_path", "data/processed/test.csv")

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")
    test_df, test_x = load_xy(test_csv, emb_dir / "test.npy")
    if "valence" not in test_df.columns or "arousal" not in test_df.columns:
        raise SystemExit("test.csv must include valence/arousal for local scoring")

    if args.strategy == "train_full_dev":
        x = np.vstack([train_x, dev_x])
        yv = np.concatenate([train_df["valence"].to_numpy(), dev_df["valence"].to_numpy()])
        ya = np.concatenate([train_df["arousal"].to_numpy(), dev_df["arousal"].to_numpy()])
    else:
        x = train_x
        yv = train_df["valence"].to_numpy()
        ya = train_df["arousal"].to_numpy()

    svr = cfg["svr"]
    model_v = train_one(x, yv, svr["C"], svr["epsilon"])
    model_a = train_one(x, ya, svr["C"], svr["epsilon"])

    pred_v = clip_va(model_v.predict(test_x))
    pred_a = clip_va(model_a.predict(test_x))
    metrics = evaluate_va(
        test_df["valence"],
        pred_v,
        test_df["arousal"],
        pred_a,
    )
    print(args.strategy, "test", format_metrics(metrics))

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "ID": test_df["id"].astype(str),
            "Valence": pred_v,
            "Arousal": pred_a,
        }
    )
    sub_path = out_dir / f"{enc}_test_{args.strategy}_submission.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")
    print(f"wrote {sub_path}")

    payload = {
        "encoder": cfg["encoder_name"],
        "strategy": args.strategy,
        "n_train": int(len(x)),
        "n_test": int(len(test_df)),
        "metrics": metrics,
        "submission": str(sub_path).replace("\\", "/"),
        "clip_range": [1.0, 9.0],
    }
    metrics_path = out_dir / f"{enc}_test_{args.strategy}.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {metrics_path}")

    if args.run_official_scoring:
        import subprocess
        import sys

        raw_dir = Path("data/raw/ROCLING-2025-ST-DSA-MST")
        ans = raw_dir / "Dataset" / "DSAMST-TestSet_ans.csv"
        staging = (out_dir / "_official_score").resolve()
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy(sub_path, staging / "submission.csv")
        shutil.copy(ans, staging / "DSAMST-TestSet_ans.csv")
        shutil.copy(raw_dir / "scoring.py", staging / "scoring.py")
        subprocess.run(
            [sys.executable, "scoring.py"],
            cwd=str(staging),
            check=True,
        )
        scores_path = staging / "scores.json"
        print(f"official scores: {scores_path.read_text(encoding='utf-8')}")
        # Persist alongside metrics for Day 4 notes.
        official = json.loads(scores_path.read_text(encoding="utf-8"))
        payload["official_scoring"] = official
        metrics_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
