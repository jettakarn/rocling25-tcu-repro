from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_CANDIDATES = ("text", "sentence", "content", "Text", "SENTENCE")
ID_CANDIDATES = ("id", "ID", "sid", "instance_id")
V_CANDIDATES = ("valence", "Valence", "V", "valence_mean", "Valence_Mean")
A_CANDIDATES = ("arousal", "Arousal", "A", "arousal_mean", "Arousal_Mean")


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    kwargs = {"encoding": "utf-8", "encoding_errors": "replace"}
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    return normalize_va_frame(df)


def normalize_va_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    text_col = _pick(cols, TEXT_CANDIDATES)
    v_col = _pick(cols, V_CANDIDATES)
    a_col = _pick(cols, A_CANDIDATES)
    id_col = _pick(cols, ID_CANDIDATES)
    if text_col is None:
        raise ValueError(f"No text column found. Columns={cols}")
    out = pd.DataFrame()
    out["id"] = df[id_col].astype(str) if id_col else [str(i) for i in range(len(df))]
    out["text"] = df[text_col].astype(str)
    if v_col is not None:
        out["valence"] = pd.to_numeric(df[v_col], errors="coerce")
    if a_col is not None:
        out["arousal"] = pd.to_numeric(df[a_col], errors="coerce")
    return out


def summarize(df: pd.DataFrame, name: str) -> str:
    lines = [f"{name}: n={len(df)}"]
    if "valence" in df.columns:
        lines.append(
            f"  valence mean={df['valence'].mean():.3f} std={df['valence'].std():.3f}"
        )
    if "arousal" in df.columns:
        lines.append(
            f"  arousal mean={df['arousal'].mean():.3f} std={df['arousal'].std():.3f}"
        )
    lines.append(f"  avg chars={df['text'].str.len().mean():.1f}")
    return "\n".join(lines)
