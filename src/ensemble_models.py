from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.embed import encoder_slug
from src.metrics import evaluate_va, format_metrics
from src.train_boost import make_model
from src.train_resnet import predict_resnet, train_resnet
from src.train_svr import load_xy, train_one


def clip_va(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 9.0)


def run_official_scoring(sub_path: Path, out_dir: Path, payload: dict, metrics_path: Path) -> None:
    raw_dir = Path("data/raw/ROCLING-2025-ST-DSA-MST")
    ans = raw_dir / "Dataset" / "DSAMST-TestSet_ans.csv"
    staging = (out_dir / "_official_score_ensemble").resolve()
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy(sub_path, staging / "submission.csv")
    shutil.copy(ans, staging / "DSAMST-TestSet_ans.csv")
    shutil.copy(raw_dir / "scoring.py", staging / "scoring.py")
    subprocess.run([sys.executable, "scoring.py"], cwd=str(staging), check=True)
    scores_path = staging / "scores.json"
    print(f"official scores: {scores_path.read_text(encoding='utf-8')}")
    payload["official_scoring"] = json.loads(scores_path.read_text(encoding="utf-8"))
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-level average ensemble (SVR + LGBM + XGB + CustomResNet)."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--strategy",
        choices=["train", "train_full_dev"],
        default="train_full_dev",
    )
    parser.add_argument("--resnet-epochs", type=int, default=40)
    parser.add_argument("--run-official-scoring", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc
    test_csv = cfg["data"].get("test_path", "data/processed/test.csv")

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")
    test_df, test_x = load_xy(test_csv, emb_dir / "test.npy")

    if args.strategy == "train_full_dev":
        x = np.vstack([train_x, dev_x])
        yv = np.concatenate([train_df["valence"].to_numpy(), dev_df["valence"].to_numpy()])
        ya = np.concatenate([train_df["arousal"].to_numpy(), dev_df["arousal"].to_numpy()])
    else:
        x = train_x
        yv = train_df["valence"].to_numpy()
        ya = train_df["arousal"].to_numpy()

    y_va = np.stack([yv, ya], axis=1)
    members: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # SVR
    svr = cfg["svr"]
    mv = train_one(x, yv, svr["C"], svr["epsilon"])
    ma = train_one(x, ya, svr["C"], svr["epsilon"])
    members["svr"] = (clip_va(mv.predict(test_x)), clip_va(ma.predict(test_x)))

    # Tree boosters
    for name in ("lgbm", "xgb", "catboost"):
        mv = make_model(name, seed)
        ma = make_model(name, seed)
        mv.fit(x, yv)
        ma.fit(x, ya)
        members[name] = (clip_va(mv.predict(test_x)), clip_va(ma.predict(test_x)))

    # CustomResNet (joint VA)
    resnet = train_resnet(
        x,
        y_va,
        seed=seed,
        device=device,
        epochs=args.resnet_epochs,
    )
    pred_r = predict_resnet(resnet, test_x, device)
    members["resnet"] = (clip_va(pred_r[:, 0]), clip_va(pred_r[:, 1]))

    per_model = {}
    for name, (pv, pa) in members.items():
        m = evaluate_va(test_df["valence"], pv, test_df["arousal"], pa)
        per_model[name] = m
        print(name, format_metrics(m))

    stack_v = np.mean([pv for pv, _ in members.values()], axis=0)
    stack_a = np.mean([pa for _, pa in members.values()], axis=0)
    ens = evaluate_va(test_df["valence"], stack_v, test_df["arousal"], stack_a)
    print("ensemble", format_metrics(ens))

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {"ID": test_df["id"].astype(str), "Valence": stack_v, "Arousal": stack_a}
    )
    sub_path = out_dir / f"{enc}_test_{args.strategy}_ensemble_submission.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")

    payload = {
        "encoder": cfg["encoder_name"],
        "strategy": args.strategy,
        "members": sorted(members.keys()),
        "n_train": int(len(x)),
        "n_test": int(len(test_df)),
        "per_model": per_model,
        "metrics": ens,
        "submission": str(sub_path).replace("\\", "/"),
        "note": "Paper Table 4 Models ensemble ≈ mean of regressor predictions.",
    }
    metrics_path = out_dir / f"{enc}_test_{args.strategy}_ensemble.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {sub_path}")
    print(f"wrote {metrics_path}")

    if args.run_official_scoring:
        run_official_scoring(sub_path, out_dir, payload, metrics_path)


if __name__ == "__main__":
    main()
