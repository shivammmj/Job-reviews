from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import PipelineConfig
    from .utils import batched, clean_text, detect_text_column, ensure_dir, read_json
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import batched, clean_text, detect_text_column, ensure_dir, read_json


@dataclass
class SentenceScorer:
    model: object
    category_names: list[str]
    centroid_matrix: np.ndarray


def _load_seed_terms(seed_path: Path) -> dict[str, list[tuple[str, float]]]:
    payload = read_json(seed_path)
    raw_categories = payload.get("categories", payload)
    categories: dict[str, list[tuple[str, float]]] = {}
    for category, entries in raw_categories.items():
        terms: list[tuple[str, float]] = []
        for entry in entries:
            if isinstance(entry, str):
                term = clean_text(entry).lower()
                score = 1.0
            else:
                term = clean_text(entry.get("term", "")).lower()
                score = float(entry.get("score", 1.0))
            if term:
                terms.append((term, score))
        if terms:
            categories[category] = terms
    if not categories:
        raise ValueError(f"No seed categories found in {seed_path}")
    return categories


def _build_sentence_scorer(
    categories: dict[str, list[tuple[str, float]]],
    config: PipelineConfig,
) -> SentenceScorer:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.seed_model_id)
    category_names = list(categories)
    centroids = []
    for category in category_names:
        terms = [term for term, _weight in categories[category]]
        weights = np.array([weight for _term, weight in categories[category]], dtype=np.float32)
        embeddings = model.encode(
            terms,
            normalize_embeddings=True,
            batch_size=config.embedding_batch_size,
            show_progress_bar=False,
        )
        centroid = np.average(embeddings, axis=0, weights=weights)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    return SentenceScorer(
        model=model,
        category_names=category_names,
        centroid_matrix=np.vstack(centroids),
    )


def _score_with_sentence_transformer(
    texts: list[str],
    scorer: SentenceScorer,
    config: PipelineConfig,
) -> pd.DataFrame:
    scores = np.zeros((len(texts), len(scorer.category_names)), dtype=np.float32)
    row_start = 0
    for batch in batched(texts, config.embedding_batch_size):
        embeddings = scorer.model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=config.embedding_batch_size,
            show_progress_bar=False,
        )
        batch_scores = np.matmul(embeddings, scorer.centroid_matrix.T)
        row_end = row_start + len(batch)
        scores[row_start:row_end] = np.maximum(batch_scores, 0.0)
        row_start = row_end
    return pd.DataFrame(scores, columns=scorer.category_names)


def _score_with_tfidf(
    texts: list[str],
    categories: dict[str, list[tuple[str, float]]],
) -> pd.DataFrame:
    category_names = list(categories)
    profiles = []
    for category in category_names:
        weighted_terms = []
        for term, weight in categories[category]:
            repeat = max(1, min(5, int(round(weight * 5))))
            weighted_terms.extend([term] * repeat)
        profiles.append(" ".join(weighted_terms))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    matrix = vectorizer.fit_transform(profiles + texts)
    profile_vectors = matrix[: len(category_names)]
    text_vectors = matrix[len(category_names) :]
    scores = cosine_similarity(text_vectors, profile_vectors)
    return pd.DataFrame(scores, columns=category_names)


def _apply_classification(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    hr_category_keys: list[str],
    threshold: float,
    tfidf_threshold: float,
    backend_used: str,
) -> pd.DataFrame:
    output = frame.copy()
    for category in scores.columns:
        output[f"score_{category}"] = scores[category].values

    available_hr_categories = [category for category in hr_category_keys if category in scores.columns]
    if available_hr_categories:
        hr_score = scores[available_hr_categories].sum(axis=1)
    else:
        hr_score = pd.Series([0.0] * len(scores), index=scores.index)

    primary_category = scores.idxmax(axis=1)
    output["primary_category"] = primary_category.values
    output["primary_category_score"] = scores.max(axis=1).values
    output["hr_score"] = hr_score.values
    threshold_used = tfidf_threshold if backend_used == "tfidf" else threshold
    output["classification"] = np.where(hr_score >= threshold_used, "HR", "Non HR")
    output["classification_threshold_used"] = threshold_used
    output["classification_backend"] = backend_used
    return output


def _classify_one_file(
    path: Path,
    output_path: Path,
    categories: dict[str, list[tuple[str, float]]],
    config: PipelineConfig,
    backend: str,
    sentence_scorer: SentenceScorer | None = None,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Review-side file not found: {path}")

    nrows = config.max_rows if config.max_rows and config.max_rows > 0 else None
    frame = pd.read_csv(path, nrows=nrows, low_memory=False)
    text_column = detect_text_column(frame.columns)
    texts = [clean_text(value) for value in frame[text_column]]

    if backend == "sentence_transformer":
        if sentence_scorer is None:
            raise ValueError("SentenceTransformer backend selected without a scorer.")
        scores = _score_with_sentence_transformer(texts, sentence_scorer, config)
        backend_used = "sentence_transformer"
    else:
        scores = _score_with_tfidf(texts, categories)
        backend_used = "tfidf"

    output = _apply_classification(
        frame=frame,
        scores=scores,
        hr_category_keys=config.hr_category_keys,
        threshold=config.classification_threshold,
        tfidf_threshold=config.tfidf_classification_threshold,
        backend_used=backend_used,
    )
    output.to_csv(output_path, index=False)
    print(
        "[classify] "
        f"{path.name}: {len(output):,} rows, "
        f"{(output['classification'] == 'Non HR').sum():,} Non HR -> {output_path}"
    )
    return output_path


def classify_review_files(
    config: PipelineConfig,
    split_paths: dict[str, Path] | None = None,
    seed_path: Path | None = None,
) -> dict[str, Path]:
    """Classify each review-side file into HR and Non HR."""
    ensure_dir(config.output_dir)
    paths = split_paths or config.split_paths
    categories = _load_seed_terms(seed_path or config.seed_path)
    backend = config.classification_backend.lower().replace("-", "_")
    if backend not in {"auto", "tfidf", "sentence_transformer"}:
        raise ValueError(f"Unsupported classification backend: {config.classification_backend}")

    sentence_scorer: SentenceScorer | None = None
    backend_used = backend
    if backend == "sentence_transformer":
        sentence_scorer = _build_sentence_scorer(categories, config)
    elif backend == "auto":
        try:
            sentence_scorer = _build_sentence_scorer(categories, config)
            backend_used = "sentence_transformer"
        except Exception as exc:
            print(f"[classify] Embedding classification failed; falling back to TF-IDF. Reason: {exc}")
            backend_used = "tfidf"

    written: dict[str, Path] = {}
    for name, path in paths.items():
        output_path = config.output_dir / f"{Path(path).stem}_classified.csv"
        written[name] = _classify_one_file(
            path=path,
            output_path=output_path,
            categories=categories,
            config=config,
            backend=backend_used,
            sentence_scorer=sentence_scorer,
        )
    return written
