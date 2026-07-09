from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import PipelineConfig
    from .utils import clean_text, ensure_dir
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import clean_text, ensure_dir


def _read_csv(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    nrows = max_rows if max_rows and max_rows > 0 else None
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _build_segment(
    df: pd.DataFrame,
    platform: str,
    prefix: str,
    source_text_column: str,
    output_text_column: str,
    sentiment_side: str,
) -> pd.DataFrame:
    text = _series(df, source_text_column).map(clean_text)
    segment = pd.DataFrame(
        {
            "review_id": [
                f"{prefix}_{idx}_{source_text_column.lower()}" for idx in df.index.astype(str)
            ],
            "source_row_id": df.index.astype(str),
            "platform": platform,
            "source_side": output_text_column,
            "sentiment_side": sentiment_side,
            "company": _series(df, "Company"),
            "rating": _series(df, "Rating")
            if platform == "glassdoor"
            else _series(df, "Overall_Rating"),
            "date": _series(df, "Date"),
            "job_title": _series(df, "Job") if platform == "glassdoor" else _series(df, "Job_Profile"),
            "review_title": _series(df, "Title")
            if platform == "glassdoor"
            else _series(df, "Review_Title"),
            "employment_type": _series(df, "Status")
            if platform == "glassdoor"
            else _series(df, "Employment_Type"),
            "location": _series(df, "Location"),
            "work_policy": _series(df, "Work_Policy"),
            output_text_column: text,
            "review_text": text,
        }
    )
    segment = segment[segment["review_text"].str.len() > 0].reset_index(drop=True)
    return segment


def split_reviews(config: PipelineConfig) -> dict[str, Path]:
    """Split raw platform files into Pros, Cons, Likes, and Dislikes files."""
    ensure_dir(config.intermediate_dir)

    glassdoor = _read_csv(config.glassdoor_input, config.max_rows)
    ambitionbox = _read_csv(config.ambitionbox_input, config.max_rows)

    outputs = {
        "pros_gd": _build_segment(
            glassdoor,
            platform="glassdoor",
            prefix="gd",
            source_text_column="Pros",
            output_text_column="Pros",
            sentiment_side="positive",
        ),
        "cons_gd": _build_segment(
            glassdoor,
            platform="glassdoor",
            prefix="gd",
            source_text_column="Cons",
            output_text_column="Cons",
            sentiment_side="negative",
        ),
        "likes_am": _build_segment(
            ambitionbox,
            platform="ambitionbox",
            prefix="am",
            source_text_column="Likes",
            output_text_column="Likes",
            sentiment_side="positive",
        ),
        "dislikes_am": _build_segment(
            ambitionbox,
            platform="ambitionbox",
            prefix="am",
            source_text_column="Dislikes",
            output_text_column="Dislikes",
            sentiment_side="negative",
        ),
    }

    written: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = config.split_paths[name]
        frame.to_csv(path, index=False)
        written[name] = path
        print(f"[split] {name}: {len(frame):,} rows -> {path}")

    return written
