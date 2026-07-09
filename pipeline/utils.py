from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator


TEXT_COLUMN_CANDIDATES = [
    "review_text",
    "Pros",
    "Cons",
    "Likes",
    "Dislikes",
    "review",
    "Review",
    "phrase",
    "text",
    "Text",
]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_default_env_files() -> None:
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent
    load_env_file(project_root / ".env")
    load_env_file(pipeline_dir / ".env")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_nlp(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_text_column(columns: Iterable[str]) -> str:
    available = list(columns)
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate
    raise ValueError(f"No review text column found. Available columns: {available}")


def batched(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value).lower()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
