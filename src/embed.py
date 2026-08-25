from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.data_loader import load_table


DEFAULT_INSTRUCT = (
    "Classify the valence and arousal of the given Chinese text "
    "on a 1-9 dimensional sentiment scale"
)


def encoder_slug(name: str) -> str:
    return name.replace("/", "__")


def with_instruct(texts: list[str], task: str) -> list[str]:
    return [f"Instruct: {task}\nQuery: {t}" for t in texts]


def with_prefix(texts: list[str], prefix: str) -> list[str]:
    return [f"{prefix}{t}" for t in texts]


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    max_length: int | None = None,
) -> np.ndarray:
    kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }
    if max_length is not None:
        model.max_seq_length = int(max_length)
    return np.asarray(model.encode(texts, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "both", "all"],
        default="both",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = SentenceTransformer(cfg["encoder_name"], device=cfg.get("device", "cuda"))
    out_dir = Path(cfg["data"]["embeddings_dir"]) / encoder_slug(cfg["encoder_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    task = cfg.get("instruct_task", DEFAULT_INSTRUCT)
    use_instruct = bool(cfg.get("use_instruct", "instruct" in str(cfg["encoder_name"]).lower()))
    embed_prefix = cfg.get("embed_prefix")  # e.g. "query: " for e5-large
    max_length = cfg.get("max_length")

    splits = []
    if args.split in {"train", "both", "all"}:
        splits.append(("train", cfg["data"]["train_path"]))
    if args.split in {"dev", "both", "all"}:
        splits.append(("dev", cfg["data"]["dev_path"]))
    if args.split in {"test", "all"}:
        test_path = cfg["data"].get("test_path", "data/processed/test.csv")
        splits.append(("test", test_path))

    for name, path in splits:
        df = load_table(path)
        texts = df["text"].tolist()
        if use_instruct:
            texts = with_instruct(texts, task)
            print(f"{name}: instruct prefix on, task={task!r}")
        elif embed_prefix:
            texts = with_prefix(texts, str(embed_prefix))
            print(f"{name}: embed_prefix={embed_prefix!r}")
        vecs = embed_texts(model, texts, cfg["batch_size"], max_length=max_length)
        out = out_dir / f"{name}.npy"
        np.save(out, vecs)
        print(f"saved {out} shape={vecs.shape}")


if __name__ == "__main__":
    main()
