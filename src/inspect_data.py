from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_table, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="CSV/TSV files to inspect")
    args = parser.parse_args()
    for p in args.paths:
        df = load_table(p)
        print(summarize(df, Path(p).name))
        print(df.head(2).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
