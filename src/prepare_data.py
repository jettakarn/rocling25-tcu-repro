from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import normalize_va_frame

RAW = Path("data/raw")
OUT = Path("data/processed")
CVAT_DIR = RAW / "ROCLING-2025-ST-DSA-MST/Dataset/ChineseEmoBank/CVAT_SD"
DEFAULT_TRAIN_FOLDS = tuple(CVAT_DIR / f"CVAT_{i}_SD.csv" for i in range(1, 6))
DEFAULT_DEV = RAW / "ROCLING-2025-ST-DSA-MST/Dataset/DSAMST-ValidationSet_ans.csv"
DEFAULT_TEST = RAW / "ROCLING-2025-ST-DSA-MST/Dataset/DSAMST-TestSet_ans.csv"


def find_files(root: Path, hints: tuple[str, ...]) -> list[Path]:
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".tsv", ".txt", ".xlsx"}:
            continue
        name = p.name.lower()
        if any(h in name or h in str(p).lower() for h in hints):
            hits.append(p)
    return sorted(hits)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
    )


def load_train_folds(fold_paths: tuple[Path, ...]) -> pd.DataFrame:
    missing = [p for p in fold_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CVAT fold files:\n  " + "\n  ".join(str(p) for p in missing))
    frames = [read_any(path) for path in fold_paths]
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"wrote {path} n={len(df)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw shared-task / EmoBank files.")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Do not write test.csv.")
    args = parser.parse_args()

    dev_path = args.dev or (DEFAULT_DEV if DEFAULT_DEV.exists() else None)
    test_path = args.test or (DEFAULT_TEST if DEFAULT_TEST.exists() else None)

    if args.train is not None:
        train = normalize_va_frame(read_any(args.train))
    elif all(p.exists() for p in DEFAULT_TRAIN_FOLDS):
        train = normalize_va_frame(load_train_folds(DEFAULT_TRAIN_FOLDS))
        print("train source: CVAT folds", *[str(p.name) for p in DEFAULT_TRAIN_FOLDS], sep="\n  ")
    else:
        cands = find_files(RAW, ("cvat", "emobank", "train"))
        print("train candidates:", *[str(p) for p in cands], sep="\n  ")
        if not cands:
            raise SystemExit("No train file found. Pass --train PATH")
        train = normalize_va_frame(read_any(cands[0]))

    if dev_path is None:
        cands = find_files(RAW, ("validationset_ans", "dev", "devel", "valid"))
        print("dev candidates:", *[str(p) for p in cands], sep="\n  ")
        if not cands:
            raise SystemExit("No labeled dev file found. Pass --dev PATH")
        dev_path = cands[0]

    if not args.skip_test and test_path is None:
        cands = find_files(RAW, ("testset_ans", "test_ans", "testset"))
        print("test candidates:", *[str(p) for p in cands], sep="\n  ")
        if cands:
            # Prefer labeled answer file when present.
            labeled = [p for p in cands if "ans" in p.name.lower()]
            test_path = labeled[0] if labeled else cands[0]

    dev = normalize_va_frame(read_any(dev_path))
    save(train, "train.csv")
    save(dev, "dev.csv")
    if not args.skip_test and test_path is not None:
        test = normalize_va_frame(read_any(test_path))
        save(test, "test.csv")


if __name__ == "__main__":
    main()
