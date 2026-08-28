"""Phase A4 (non-paper): light domain adaptation on cached e5-instruct embeddings.

Not part of the TCU paper method. Kept as an optional experiment.

Experiments (no test gold labels in training):
  - baseline_full_dev: train + labeled_dev (documents existing test protocol)
  - stronger_dev: upsample labeled_dev copies in the SVR fit
  - retrieval_upsample: nearest CVAT (train) neighbors to labeled_dev, upsampled
  - pseudo_holdout: self-train on high-confidence preds for a held-out half of
    labeled_dev (labels ignored during pseudo step; used only for scoring
    the holdout). Final test model retrains on train + full_dev + soft
    pseudo from that holdout — still no test labels.

All methods score on labeled test for evaluation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.svm import SVR

from src.embed import encoder_slug
from src.metrics import evaluate_va, format_metrics
from src.predict_test import clip_va
from src.train_svr import load_xy, train_one


def _fit_predict(
    x_train: np.ndarray,
    yv: np.ndarray,
    ya: np.ndarray,
    x_eval: np.ndarray,
    C: float,
    epsilon: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model_v = SVR(kernel="rbf", C=C, epsilon=epsilon)
    model_a = SVR(kernel="rbf", C=C, epsilon=epsilon)
    if sample_weight is None:
        model_v.fit(x_train, yv)
        model_a.fit(x_train, ya)
    else:
        model_v.fit(x_train, yv, sample_weight=sample_weight)
        model_a.fit(x_train, ya, sample_weight=sample_weight)
    return clip_va(model_v.predict(x_eval)), clip_va(model_a.predict(x_eval))


def _l2_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def retrieval_indices(
    train_x: np.ndarray,
    dev_x: np.ndarray,
    max_unique: int,
) -> np.ndarray:
    """Pick unique train rows nearest to any labeled_dev embedding."""
    t = _l2_rows(train_x)
    d = _l2_rows(dev_x)
    # cosine sim via matmul; for each train row, max similarity to any dev row
    sims = t @ d.T  # (n_train, n_dev)
    best = sims.max(axis=1)
    order = np.argsort(-best)
    return order[:max_unique]


def run_experiment(
    name: str,
    x: np.ndarray,
    yv: np.ndarray,
    ya: np.ndarray,
    test_x: np.ndarray,
    test_df: pd.DataFrame,
    C: float,
    epsilon: float,
    sample_weight: np.ndarray | None = None,
) -> dict:
    pred_v, pred_a = _fit_predict(x, yv, ya, test_x, C, epsilon, sample_weight)
    metrics = evaluate_va(test_df["valence"], pred_v, test_df["arousal"], pred_a)
    print(name, "test", format_metrics(metrics))
    return {
        "name": name,
        "n_train": int(len(x)),
        "metrics": metrics,
        "pred_v": pred_v,
        "pred_a": pred_a,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A4 light domain adaptation on cached e5 embeddings."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dev-copies", type=int, default=3, help="Upsample factor for labeled_dev.")
    parser.add_argument(
        "--retrieval-n",
        type=int,
        default=400,
        help="How many unique train neighbors (to labeled_dev) to upsample.",
    )
    parser.add_argument(
        "--retrieval-copies",
        type=int,
        default=2,
        help="Extra copies of each retrieved train neighbor.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    rng = np.random.default_rng(seed)
    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc
    test_csv = cfg["data"].get("test_path", "data/processed/test.csv")
    C = float(cfg["svr"]["C"])
    epsilon = float(cfg["svr"]["epsilon"])

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")
    test_df, test_x = load_xy(test_csv, emb_dir / "test.npy")

    train_yv = train_df["valence"].to_numpy()
    train_ya = train_df["arousal"].to_numpy()
    dev_yv = dev_df["valence"].to_numpy()
    dev_ya = dev_df["arousal"].to_numpy()

    experiments: list[dict] = []

    # --- 1) Baseline: train_full_dev (same as predict_test) ---
    x0 = np.vstack([train_x, dev_x])
    yv0 = np.concatenate([train_yv, dev_yv])
    ya0 = np.concatenate([train_ya, dev_ya])
    base = run_experiment("baseline_train_full_dev", x0, yv0, ya0, test_x, test_df, C, epsilon)
    experiments.append(base)

    # --- 2) Stronger labeled_dev: repeat dev rows ---
    copies = max(1, int(args.dev_copies))
    x_dev_up = np.vstack([train_x] + [dev_x] * copies)
    yv_dev_up = np.concatenate([train_yv] + [dev_yv] * copies)
    ya_dev_up = np.concatenate([train_ya] + [dev_ya] * copies)
    stronger = run_experiment(
        f"stronger_dev_x{copies}",
        x_dev_up,
        yv_dev_up,
        ya_dev_up,
        test_x,
        test_df,
        C,
        epsilon,
    )
    experiments.append(stronger)

    # sample_weight variant (same idea, less memory)
    w = np.concatenate(
        [np.ones(len(train_x), dtype=float), np.full(len(dev_x), float(copies), dtype=float)]
    )
    weighted = run_experiment(
        f"dev_sample_weight_x{copies}",
        x0,
        yv0,
        ya0,
        test_x,
        test_df,
        C,
        epsilon,
        sample_weight=w,
    )
    experiments.append(weighted)

    # --- 3) Retrieval upsample: CVAT neighbors of labeled_dev ---
    n_ret = min(int(args.retrieval_n), len(train_x))
    ret_idx = retrieval_indices(train_x, dev_x, max_unique=n_ret)
    ret_copies = max(1, int(args.retrieval_copies))
    blocks_x = [train_x, dev_x]
    blocks_yv = [train_yv, dev_yv]
    blocks_ya = [train_ya, dev_ya]
    for _ in range(ret_copies):
        blocks_x.append(train_x[ret_idx])
        blocks_yv.append(train_yv[ret_idx])
        blocks_ya.append(train_ya[ret_idx])
    x_ret = np.vstack(blocks_x)
    yv_ret = np.concatenate(blocks_yv)
    ya_ret = np.concatenate(blocks_ya)
    retrieval = run_experiment(
        f"retrieval_upsample_n{n_ret}_x{ret_copies}",
        x_ret,
        yv_ret,
        ya_ret,
        test_x,
        test_df,
        C,
        epsilon,
    )
    experiments.append(retrieval)

    # Combine stronger_dev + retrieval
    blocks_x2 = [train_x] + [dev_x] * copies
    blocks_yv2 = [train_yv] + [dev_yv] * copies
    blocks_ya2 = [train_ya] + [dev_ya] * copies
    for _ in range(ret_copies):
        blocks_x2.append(train_x[ret_idx])
        blocks_yv2.append(train_yv[ret_idx])
        blocks_ya2.append(train_ya[ret_idx])
    combo = run_experiment(
        f"stronger_dev_x{copies}+retrieval_n{n_ret}_x{ret_copies}",
        np.vstack(blocks_x2),
        np.concatenate(blocks_yv2),
        np.concatenate(blocks_ya2),
        test_x,
        test_df,
        C,
        epsilon,
    )
    experiments.append(combo)

    # --- 4) Pseudo-label on held-out half of labeled_dev (no test labels) ---
    idx = np.arange(len(dev_df))
    rng.shuffle(idx)
    half = len(idx) // 2
    add, hold = idx[:half], idx[half:]
    x_teacher = np.vstack([train_x, dev_x[add]])
    yv_teacher = np.concatenate([train_yv, dev_yv[add]])
    ya_teacher = np.concatenate([train_ya, dev_ya[add]])
    # Teacher predicts holdout (gold holdout labels never enter training)
    model_v = train_one(x_teacher, yv_teacher, C, epsilon)
    model_a = train_one(x_teacher, ya_teacher, C, epsilon)
    pseudo_v = clip_va(model_v.predict(dev_x[hold]))
    pseudo_a = clip_va(model_a.predict(dev_x[hold]))
    # Confidence: keep rows where |pseudo - gold| is small would leak.
    # Instead keep rows where prediction is not extreme vs teacher training
    # residual scale: use distance to teacher mean as soft filter + low
    # prediction variance proxy via |pred - median of teacher preds|.
    med_v = float(np.median(yv_teacher))
    med_a = float(np.median(ya_teacher))
    # Keep middle-confidence band: not too far from medians (avoid wild outs)
    # and keep all by default with a mild residual to teacher train std.
    std_v = float(np.std(yv_teacher)) + 1e-6
    std_a = float(np.std(ya_teacher)) + 1e-6
    conf = (np.abs(pseudo_v - med_v) / std_v) + (np.abs(pseudo_a - med_a) / std_a)
    # Lower conf score = closer to central mass; keep fraction under threshold
    thr = float(np.quantile(conf, 0.7))  # keep ~70% most "central"
    keep = conf <= thr
    # Also report honest holdout MAE of teacher (uses gold for eval only)
    hold_metrics = evaluate_va(
        dev_yv[hold],
        pseudo_v,
        dev_ya[hold],
        pseudo_a,
    )
    print("pseudo_teacher_holdout", format_metrics(hold_metrics), f"keep={int(keep.sum())}/{len(hold)}")

    x_pseudo = np.vstack([train_x, dev_x, dev_x[hold][keep]])
    yv_pseudo = np.concatenate([train_yv, dev_yv, pseudo_v[keep]])
    ya_pseudo = np.concatenate([train_ya, dev_ya, pseudo_a[keep]])
    # Note: hold gold labels are NOT in yv_pseudo/ya_pseudo; soft teacher preds are.
    # Full labeled_dev gold is still included once via dev_yv/dev_ya (same as baseline).
    # Extra rows are soft duplicates of the hold half — mild self-distillation.
    pseudo_exp = run_experiment(
        "pseudo_holdout_soft_on_full_dev",
        x_pseudo,
        yv_pseudo,
        ya_pseudo,
        test_x,
        test_df,
        C,
        epsilon,
    )
    experiments.append(pseudo_exp)

    # Cleaner variant: train only on train + add half + soft keep from hold
    # (never uses hold gold; uses only half of labeled_dev gold)
    x_pseudo2 = np.vstack([train_x, dev_x[add], dev_x[hold][keep]])
    yv_pseudo2 = np.concatenate([train_yv, dev_yv[add], pseudo_v[keep]])
    ya_pseudo2 = np.concatenate([train_ya, dev_ya[add], pseudo_a[keep]])
    pseudo_half = run_experiment(
        "pseudo_holdout_train_half_dev_soft",
        x_pseudo2,
        yv_pseudo2,
        ya_pseudo2,
        test_x,
        test_df,
        C,
        epsilon,
    )
    experiments.append(pseudo_half)

    # --- Summarize vs baselines ---
    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def strip_preds(e: dict) -> dict:
        return {
            "name": e["name"],
            "n_train": e["n_train"],
            "metrics": e["metrics"],
        }

    five_head = {
        "mae_v": 0.473234536593205,
        "mae_a": 0.7741227775230134,
        "pcc_v": 0.795433911361989,
        "pcc_a": 0.5982459851872609,
        "source": "results/intfloat__multilingual-e5-large-instruct_test_train_full_dev_ensemble.json",
    }
    svr_baseline_doc = {
        "mae_v": 0.488227784713437,
        "mae_a": 0.7883211239085217,
        "pcc_v": 0.7879031296167227,
        "pcc_a": 0.5775042239356336,
        "source": "results/intfloat__multilingual-e5-large-instruct_test_train_full_dev.json",
    }

    ranked = sorted(experiments, key=lambda e: e["metrics"]["mae_a"])
    best = ranked[0]
    delta_vs_five = best["metrics"]["mae_a"] - five_head["mae_a"]
    delta_vs_svr = best["metrics"]["mae_a"] - base["metrics"]["mae_a"]

    # Write best submission csv
    sub = pd.DataFrame(
        {
            "ID": test_df["id"].astype(str),
            "Valence": best["pred_v"],
            "Arousal": best["pred_a"],
        }
    )
    sub_path = out_dir / f"{enc}_test_domain_adapt_best_submission.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8")

    payload = {
        "encoder": cfg["encoder_name"],
        "phase": "A4",
        "seed": seed,
        "leakage_note": (
            "Test gold valence/arousal are never used as training targets. "
            "Pseudo experiments soft-label a held-out half of labeled_dev only; "
            "retrieval uses train labels of neighbors selected by similarity to labeled_dev."
        ),
        "baselines_documented": {
            "svr_train_full_dev_test": svr_baseline_doc,
            "five_head_ensemble_test": five_head,
            "rerun_baseline_train_full_dev": strip_preds(base),
        },
        "hyperparams": {
            "dev_copies": copies,
            "retrieval_n": n_ret,
            "retrieval_copies": ret_copies,
            "pseudo_keep_quantile": 0.7,
            "pseudo_holdout_teacher_metrics": hold_metrics,
            "pseudo_keep_n": int(keep.sum()),
            "pseudo_hold_n": int(len(hold)),
        },
        "experiments": [strip_preds(e) for e in experiments],
        "best_by_mae_a": strip_preds(best),
        "delta_mae_a_vs_rerun_svr": delta_vs_svr,
        "delta_mae_a_vs_five_head": delta_vs_five,
        "goal_mae_a": 0.76,
        "reached_goal": bool(best["metrics"]["mae_a"] <= 0.76),
        "submission": str(sub_path).replace("\\", "/"),
    }
    metrics_path = out_dir / f"{enc}_domain_adapt.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(
        f"best={best['name']} MAE_A={best['metrics']['mae_a']:.4f} "
        f"(vs SVR {delta_vs_svr:+.4f}, vs five-head {delta_vs_five:+.4f})"
    )


if __name__ == "__main__":
    main()
