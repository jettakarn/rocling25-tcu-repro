from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import yaml

try:
    from catboost import CatBoostRegressor
except ImportError:  # optional until installed
    CatBoostRegressor = None  # type: ignore

from src.metrics import evaluate_va, format_metrics
from src.train_svr import build_split


def make_model(name: str, seed: int):
    if name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            random_state=seed,
            verbosity=-1,
        )
    if name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )
    if name == "catboost":
        if CatBoostRegressor is None:
            raise SystemExit("catboost not installed. pip install catboost")
        return CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGBM/XGB/CatBoost on cached embeddings.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "catboost", "both", "all"],
        default="both",
    )
    parser.add_argument(
        "--strategy",
        choices=["train", "train_half_dev", "train_full_dev"],
        default="train_half_dev",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    x, yv, ya, dev_df, dev_x, eval_mask, enc = build_split(cfg, args.strategy)
    if args.strategy == "train_full_dev":
        print("warning: train_full_dev evaluates on the same labeled dev set (optimistic).")

    if args.model == "both":
        models = ["lgbm", "xgb"]
    elif args.model == "all":
        models = ["lgbm", "xgb", "catboost"]
    else:
        models = [args.model]
    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in models:
        mv = make_model(name, seed)
        ma = make_model(name, seed)
        mv.fit(x, yv)
        ma.fit(x, ya)
        pred_v = mv.predict(dev_x)
        pred_a = ma.predict(dev_x)
        metrics = evaluate_va(
            dev_df.loc[eval_mask, "valence"],
            pred_v[eval_mask],
            dev_df.loc[eval_mask, "arousal"],
            pred_a[eval_mask],
        )
        print(name, args.strategy, format_metrics(metrics))
        out = out_dir / f"{enc}_{name}_{args.strategy}.json"
        payload = {
            "encoder": cfg["encoder_name"],
            "model": name,
            "strategy": args.strategy,
            "n_train": int(len(x)),
            "n_eval": int(eval_mask.sum()),
            "metrics": metrics,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
