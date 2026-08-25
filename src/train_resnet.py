from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.custom_resnet import CustomResNet
from src.embed import encoder_slug
from src.metrics import evaluate_va, format_metrics
from src.train_svr import load_xy


def build_xy(cfg, strategy: str):
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    enc = encoder_slug(cfg["encoder_name"])
    emb_dir = Path(cfg["data"]["embeddings_dir"]) / enc

    train_df, train_x = load_xy(cfg["data"]["train_path"], emb_dir / "train.npy")
    dev_df, dev_x = load_xy(cfg["data"]["dev_path"], emb_dir / "dev.npy")

    if strategy == "train":
        x = train_x
        y = np.stack([train_df["valence"].to_numpy(), train_df["arousal"].to_numpy()], axis=1)
        eval_mask = np.ones(len(dev_df), dtype=bool)
    elif strategy == "train_full_dev":
        x = np.vstack([train_x, dev_x])
        y = np.stack(
            [
                np.concatenate([train_df["valence"], dev_df["valence"]]),
                np.concatenate([train_df["arousal"], dev_df["arousal"]]),
            ],
            axis=1,
        )
        eval_mask = np.ones(len(dev_df), dtype=bool)
    else:
        idx = np.arange(len(dev_df))
        rng.shuffle(idx)
        half = len(idx) // 2
        add, hold = idx[:half], idx[half:]
        x = np.vstack([train_x, dev_x[add]])
        y = np.stack(
            [
                np.concatenate(
                    [train_df["valence"].to_numpy(), dev_df.iloc[add]["valence"].to_numpy()]
                ),
                np.concatenate(
                    [train_df["arousal"].to_numpy(), dev_df.iloc[add]["arousal"].to_numpy()]
                ),
            ],
            axis=1,
        )
        eval_mask = np.zeros(len(dev_df), dtype=bool)
        eval_mask[hold] = True

    return x, y, train_df, train_x, dev_df, dev_x, eval_mask, enc


def train_resnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    device: str,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 512,
) -> CustomResNet:
    torch.manual_seed(seed)
    model = CustomResNet(in_dim=x.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_resnet(model: CustomResNet, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    tensor = torch.tensor(x, dtype=torch.float32)
    for i in range(0, len(tensor), 256):
        batch = tensor[i : i + 256].to(device)
        out.append(model(batch).cpu().numpy())
    return np.vstack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CustomResNet on cached embeddings.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--strategy",
        choices=["train", "train_half_dev", "train_full_dev"],
        default="train_half_dev",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, y, _, _, dev_df, dev_x, eval_mask, enc = build_xy(cfg, args.strategy)
    if args.strategy == "train_full_dev":
        print("warning: train_full_dev evaluates on the same labeled dev set (optimistic).")

    model = train_resnet(x, y, seed=int(cfg["seed"]), device=device, epochs=args.epochs)
    pred = predict_resnet(model, dev_x, device)
    metrics = evaluate_va(
        dev_df.loc[eval_mask, "valence"],
        pred[eval_mask, 0],
        dev_df.loc[eval_mask, "arousal"],
        pred[eval_mask, 1],
    )
    print("resnet", args.strategy, format_metrics(metrics))

    out_dir = Path(cfg["data"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{enc}_resnet_{args.strategy}.json"
    payload = {
        "encoder": cfg["encoder_name"],
        "model": "custom_resnet",
        "strategy": args.strategy,
        "n_train": int(len(x)),
        "n_eval": int(eval_mask.sum()),
        "epochs": args.epochs,
        "metrics": metrics,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
