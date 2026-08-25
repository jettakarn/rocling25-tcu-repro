from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from src.embed import encoder_slug
from src.metrics import format_metrics, mae, pcc
from src.train_svr import load_xy, train_one


def build_split(cfg, strategy: str):
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")

    if strategy == "train":
        x = train_x
        yv = train_df["valence"].to_numpy()
        ya = train_df["arousal"].to_numpy()
        eval_mask = np.ones(len(dev_df), dtype=bool)
    elif strategy == "train_full_dev":
        x = np.vstack([train_x, dev_x])
        yv = np.concatenate([train_df["valence"], dev_df["valence"]])
        ya = np.concatenate([train_df["arousal"], dev_df["arousal"]])
        eval_mask = np.ones(len(dev_df), dtype=bool)
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

    return x, yv, ya, dev_df, dev_x, eval_mask, enc


def sweep_dim(x, y, x_eval, y_true, Cs, epsilons, label: str):
    rows = []
    best = None
    for C, eps in itertools.product(Cs, epsilons):
        model = train_one(x, y, C, eps)
        pred = model.predict(x_eval)
        row = {
            "C": C,
            "epsilon": eps,
            "mae": mae(y_true, pred),
            "pcc": pcc(y_true, pred),
        }
        rows.append(row)
        print(f"{label} C={C:<4} eps={eps:<4}  MAE={row['mae']:.3f}  PCC={row['pcc']:.3f}")
        if best is None or row["mae"] < best["mae"]:
            best = {**row, "pred": pred}
    return rows, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search SVR C/epsilon per VA head.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--strategy",
        choices=["train", "train_half_dev", "train_full_dev"],
        default="train_half_dev",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    x, yv, ya, dev_df, dev_x, eval_mask, enc = build_split(cfg, args.strategy)
    x_eval = dev_x[eval_mask]
    yv_true = dev_df.loc[eval_mask, "valence"].to_numpy()
    ya_true = dev_df.loc[eval_mask, "arousal"].to_numpy()

    Cs = [1.0, 3.0, 10.0, 30.0]
    epsilons = [0.05, 0.1, 0.2]

    print("=== sweep valence ===")
    grid_v, best_v = sweep_dim(x, yv, x_eval, yv_true, Cs, epsilons, "V")
    print("=== sweep arousal ===")
    grid_a, best_a = sweep_dim(x, ya, x_eval, ya_true, Cs, epsilons, "A")

    combined = {
        "mae_v": best_v["mae"],
        "mae_a": best_a["mae"],
        "pcc_v": best_v["pcc"],
        "pcc_a": best_a["pcc"],
    }
    print(
        "combined_best",
        format_metrics(combined),
        f"C_v={best_v['C']} eps_v={best_v['epsilon']} "
        f"C_a={best_a['C']} eps_a={best_a['epsilon']}",
    )

    # Paper defaults for reference
    base_v = next(r for r in grid_v if r["C"] == 10.0 and r["epsilon"] == 0.2)
    base_a = next(r for r in grid_a if r["C"] == 10.0 and r["epsilon"] == 0.2)
    print(
        "paper_defaults",
        format_metrics(
            {
                "mae_v": base_v["mae"],
                "mae_a": base_a["mae"],
                "pcc_v": base_v["pcc"],
                "pcc_a": base_a["pcc"],
            }
        ),
    )

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{enc}_{args.strategy}_tune.json"
    payload = {
        "encoder": cfg["encoder_name"],
        "strategy": args.strategy,
        "grid_v": [{k: v for k, v in r.items()} for r in grid_v],
        "grid_a": [{k: v for k, v in r.items()} for r in grid_a],
        "best_v": {k: v for k, v in best_v.items() if k != "pred"},
        "best_a": {k: v for k, v in best_a.items() if k != "pred"},
        "combined_best": {
            "C_v": best_v["C"],
            "epsilon_v": best_v["epsilon"],
            "C_a": best_a["C"],
            "epsilon_a": best_a["epsilon"],
            **combined,
        },
        "paper_defaults": {
            "C": 10.0,
            "epsilon": 0.2,
            "mae_v": base_v["mae"],
            "mae_a": base_a["mae"],
            "pcc_v": base_v["pcc"],
            "pcc_a": base_a["pcc"],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
