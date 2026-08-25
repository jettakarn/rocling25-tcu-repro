from __future__ import annotations

"""Average SVR predictions across multiple cached encoders (paper Table 4 Encoders spirit).

Requires each encoder's train/dev/test .npy under data/embeddings/<slug>/.
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Multi-encoder mean of SVR heads.")
    parser.add_argument("--data-config", default="configs/experiment.yaml")
    parser.add_argument(
        "--encoders",
        nargs="+",
        required=True,
        help="HF encoder names that already have cached embeddings.",
    )
    parser.add_argument(
        "--strategy",
        choices=["train", "train_full_dev"],
        default="train_full_dev",
    )
    args = parser.parse_args()

    with open(args.data_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_path = cfg["data"]["train_path"]
    dev_path = cfg["data"]["dev_path"]
    test_path = cfg["data"].get("test_path", "data/processed/test.csv")
    emb_root = Path(cfg["data"]["embeddings_dir"])
    svr = cfg["svr"]

    train_df = None
    test_df = None
    member_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for enc_name in args.encoders:
        slug = encoder_slug(enc_name)
        emb_dir = emb_root / slug
        tdf, train_x = load_xy(train_path, emb_dir / "train.npy")
        ddf, dev_x = load_xy(dev_path, emb_dir / "dev.npy")
        tedf, test_x = load_xy(test_path, emb_dir / "test.npy")
        train_df, test_df = tdf, tedf

        if args.strategy == "train_full_dev":
            x = np.vstack([train_x, dev_x])
            yv = np.concatenate([tdf["valence"].to_numpy(), ddf["valence"].to_numpy()])
            ya = np.concatenate([tdf["arousal"].to_numpy(), ddf["arousal"].to_numpy()])
        else:
            x = train_x
            yv = tdf["valence"].to_numpy()
            ya = tdf["arousal"].to_numpy()

        mv = train_one(x, yv, svr["C"], svr["epsilon"])
        ma = train_one(x, ya, svr["C"], svr["epsilon"])
        pv, pa = clip_va(mv.predict(test_x)), clip_va(ma.predict(test_x))
        member_preds[slug] = (pv, pa)
        m = evaluate_va(tedf["valence"], pv, tedf["arousal"], pa)
        print(slug, format_metrics(m))

    assert test_df is not None
    stack_v = np.mean([pv for pv, _ in member_preds.values()], axis=0)
    stack_a = np.mean([pa for _, pa in member_preds.values()], axis=0)
    ens = evaluate_va(test_df["valence"], stack_v, test_df["arousal"], stack_a)
    print("encoder_ensemble", format_metrics(ens))

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "+".join(encoder_slug(e).split("__")[-1] for e in args.encoders)
    sub = pd.DataFrame(
        {"ID": test_df["id"].astype(str), "Valence": stack_v, "Arousal": stack_a}
    )
    sub_path = out_dir / f"encoder_ensemble_{tag}_test_{args.strategy}_submission.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    payload = {
        "encoders": args.encoders,
        "strategy": args.strategy,
        "per_encoder": {
            k: evaluate_va(test_df["valence"], pv, test_df["arousal"], pa)
            for k, (pv, pa) in member_preds.items()
        },
        "metrics": ens,
        "submission": str(sub_path).replace("\\", "/"),
    }
    out = out_dir / f"encoder_ensemble_{tag}_test_{args.strategy}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {sub_path}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
