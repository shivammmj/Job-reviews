from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import PipelineConfig
    from .utils import clean_text, unique_preserve_order, write_json
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import clean_text, unique_preserve_order, write_json


BASE_SEED_CATEGORIES: dict[str, list[str]] = {
    "workplace_environment_culture_relationships": [
        "community",
        "co-worker",
        "culture",
        "workplace environment",
        "collaboration",
        "friendly",
        "atmosphere",
    ],
    "economic_psychological_benefits": [
        "work-life balance",
        "wellbeing",
        "monetary benefits",
        "goodies",
    ],
    "job_satisfaction": [
        "meaningful work",
        "job satisfaction",
        "gratification",
    ],
    "justice": [
        "fair",
        "justice",
        "just",
    ],
    "employee_brand_identification": [
        "feel like family",
        "we",
        "our",
        "second home",
    ],
    "brand_reputation": [
        "great brand",
        "respected brand",
        "valued brand",
        "good brand name",
    ],
    "product_belief": [
        "products",
        "quality",
    ],
    "employee_brand_love": [
        "love",
        "like",
        "passionate",
    ],
    "inner_self_expressive_brands": [
        "share my belief",
        "values align with",
        "my personality",
    ],
    "brand_authenticity": [
        "genuine",
        "honest",
        "transparency",
        "trust",
        "accountability",
        "authentic",
    ],
    "social_self_expressive_brands": [
        "feel important",
        "people think",
        "tell people",
    ],
}


CATEGORY_SEED_SOURCE = "category_keyword_extension_final.py categories"
EXPANSION_SOURCE = "category_keyword_extension_final.py embedding centroid approach"


def _read_keywords(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")
    frame = pd.read_csv(path)
    for column in ["keyword", "term", "words", "phrase"]:
        if column in frame.columns:
            return unique_preserve_order(frame[column].map(clean_text))
    raise ValueError(f"No keyword column found in {path}. Columns: {list(frame.columns)}")


def _new_seed_payload(backend: str, model_id: str, threshold: float) -> dict:
    return {
        "_meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model_id": model_id,
            "threshold": threshold,
            "base_category_count": len(BASE_SEED_CATEGORIES),
        },
        "categories": {category: [] for category in BASE_SEED_CATEGORIES},
    }


def _add_seed(
    category_terms: dict[str, dict[str, dict]],
    category: str,
    term: str,
    score: float,
    source: str,
) -> None:
    cleaned = clean_text(term).lower()
    if not cleaned:
        return
    existing = category_terms[category].get(cleaned)
    if existing is None or score > existing["score"]:
        category_terms[category][cleaned] = {
            "term": cleaned,
            "score": round(float(score), 4),
            "source": source,
        }


def _finalize_payload(
    payload: dict,
    category_terms: dict[str, dict[str, dict]],
    top_n_per_category: int,
) -> dict:
    for category, term_map in category_terms.items():
        terms = sorted(term_map.values(), key=lambda item: (-item["score"], item["term"]))
        payload["categories"][category] = terms[:top_n_per_category]
    return payload


def _base_category_terms() -> dict[str, dict[str, dict]]:
    category_terms = {category: {} for category in BASE_SEED_CATEGORIES}
    for category, seeds in BASE_SEED_CATEGORIES.items():
        for seed in seeds:
            _add_seed(category_terms, category, seed, 1.0, "category_keyword_extension_seed")
    return category_terms


def _create_paper_exact_seed() -> dict:
    payload = _new_seed_payload("paper_exact", CATEGORY_SEED_SOURCE, 1.0)
    payload["_meta"]["seed_source"] = CATEGORY_SEED_SOURCE
    payload["_meta"]["uses_corpus_expansion"] = False
    return _finalize_payload(
        payload,
        _base_category_terms(),
        top_n_per_category=max(len(terms) for terms in BASE_SEED_CATEGORIES.values()),
    )


def _create_paper_embedding_expanded_seed(
    keywords: list[str],
    model_id: str,
    threshold: float,
    batch_size: int,
    top_n_per_category: int,
) -> dict:
    from sentence_transformers import SentenceTransformer

    payload = _new_seed_payload("paper_embedding_expanded", model_id, threshold)
    payload["_meta"]["seed_source"] = CATEGORY_SEED_SOURCE
    payload["_meta"]["expansion_source"] = EXPANSION_SOURCE
    payload["_meta"]["uses_corpus_expansion"] = True
    payload["_meta"]["corpus_keyword_count"] = len(keywords)

    category_terms = _base_category_terms()
    category_names = list(BASE_SEED_CATEGORIES)
    model = SentenceTransformer(model_id)

    centroids = []
    for category in category_names:
        seed_embeddings = model.encode(
            BASE_SEED_CATEGORIES[category],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        centroid = np.mean(seed_embeddings, axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    centroid_matrix = np.vstack(centroids)

    keyword_embeddings = model.encode(
        keywords,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    similarities = np.matmul(keyword_embeddings, centroid_matrix.T)

    for keyword, scores in zip(keywords, similarities):
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if best_score >= threshold:
            _add_seed(
                category_terms,
                category_names[best_index],
                keyword,
                best_score,
                "keyword_embedding_similarity",
            )

    return _finalize_payload(payload, category_terms, top_n_per_category)


def create_seed_json(config: PipelineConfig, keywords_path: Path | None = None) -> Path:
    """Create seed.json from paper terms, optionally expanding with corpus keywords."""
    backend = config.seed_backend.lower().replace("-", "_")
    if backend not in {"paper_embedding_expanded", "paper_exact", "auto", "tfidf", "sentence_transformer"}:
        raise ValueError(f"Unsupported seed backend: {config.seed_backend}")

    if backend == "paper_exact":
        print("[seed] Writing exact category_keyword_extension_final.py seed terms only")
        payload = _create_paper_exact_seed()
    else:
        if backend in {"auto", "sentence_transformer"}:
            backend = "paper_embedding_expanded"
        if backend == "tfidf":
            print(
                "[seed] TF-IDF seed expansion is not used in the linked approach; "
                "using paper_embedding_expanded instead."
            )
            backend = "paper_embedding_expanded"
        path = keywords_path or config.keywords_path
        keywords = _read_keywords(path)
        if not keywords:
            raise ValueError("No keywords available for seed expansion.")
        print(
            "[seed] Expanding category_keyword_extension_final.py seeds with corpus keywords using "
            f"{EXPANSION_SOURCE}; keywords={len(keywords):,}, threshold={config.seed_similarity_threshold}"
        )
        payload = _create_paper_embedding_expanded_seed(
            keywords,
            model_id=config.seed_model_id,
            threshold=config.seed_similarity_threshold,
            batch_size=config.embedding_batch_size,
            top_n_per_category=config.seed_top_n_per_category,
        )
    write_json(config.seed_path, payload)
    print(f"[seed] seed.json -> {config.seed_path}")
    return config.seed_path
