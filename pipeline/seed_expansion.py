from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import PipelineConfig
    from .utils import clean_text, unique_preserve_order, write_json
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import clean_text, unique_preserve_order, write_json


BASE_SEED_CATEGORIES: dict[str, list[str]] = {
    "workplace_environment_culture_relationships": [
        "work culture",
        "culture",
        "team",
        "people",
        "manager",
        "management",
        "leadership",
        "colleagues",
        "supportive",
        "toxic",
        "politics",
        "communication",
        "employee friendly",
    ],
    "economic_psychological_benefits": [
        "salary",
        "pay",
        "compensation",
        "benefits",
        "bonus",
        "incentive",
        "insurance",
        "perks",
        "mental health",
        "stress",
        "pressure",
        "work life balance",
    ],
    "career_growth_learning_development": [
        "career growth",
        "promotion",
        "learning",
        "training",
        "skill development",
        "mentorship",
        "growth opportunity",
        "internal mobility",
        "performance review",
        "appraisal",
    ],
    "job_security_stability": [
        "job security",
        "stability",
        "layoff",
        "attrition",
        "contract",
        "permanent",
        "notice period",
        "bench",
        "business continuity",
    ],
    "work_life_balance_flexibility": [
        "work life balance",
        "flexible hours",
        "remote work",
        "hybrid",
        "work from home",
        "shifts",
        "weekend",
        "overtime",
        "leave policy",
    ],
    "role_work_product_delivery": [
        "project",
        "product",
        "technology",
        "technical work",
        "client",
        "customer",
        "process",
        "tools",
        "innovation",
        "quality",
        "delivery",
    ],
    "facilities_location_operations": [
        "office",
        "location",
        "transport",
        "food",
        "cafeteria",
        "infrastructure",
        "parking",
        "facility",
        "workplace",
        "campus",
    ],
}


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
            _add_seed(category_terms, category, seed, 1.0, "base_seed")
    return category_terms


def _create_tfidf_seed(
    keywords: list[str],
    threshold: float,
    top_n_per_category: int,
) -> dict:
    effective_threshold = threshold if threshold < 0.25 else 0.05
    payload = _new_seed_payload("tfidf", "sklearn.TfidfVectorizer", effective_threshold)
    category_terms = _base_category_terms()

    category_names = list(BASE_SEED_CATEGORIES)
    profiles = [" ".join(BASE_SEED_CATEGORIES[category]) for category in category_names]
    documents = profiles + keywords
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    matrix = vectorizer.fit_transform(documents)
    category_vectors = matrix[: len(category_names)]
    keyword_vectors = matrix[len(category_names) :]
    similarities = cosine_similarity(keyword_vectors, category_vectors)

    for keyword, scores in zip(keywords, similarities):
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if best_score >= effective_threshold:
            _add_seed(
                category_terms,
                category_names[best_index],
                keyword,
                best_score,
                "keyword_tfidf_similarity",
            )

    return _finalize_payload(payload, category_terms, top_n_per_category)


def _create_sentence_transformer_seed(
    keywords: list[str],
    model_id: str,
    threshold: float,
    batch_size: int,
    top_n_per_category: int,
) -> dict:
    from sentence_transformers import SentenceTransformer

    payload = _new_seed_payload("sentence_transformer", model_id, threshold)
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
    """Expand base category seed terms with corpus keywords."""
    path = keywords_path or config.keywords_path
    keywords = _read_keywords(path)
    if not keywords:
        raise ValueError("No keywords available for seed expansion.")

    backend = config.seed_backend.lower().replace("-", "_")
    if backend not in {"auto", "tfidf", "sentence_transformer"}:
        raise ValueError(f"Unsupported seed backend: {config.seed_backend}")

    print(f"[seed] Expanding {len(keywords):,} keywords with backend={backend}")
    if backend == "tfidf":
        payload = _create_tfidf_seed(
            keywords,
            threshold=config.seed_similarity_threshold,
            top_n_per_category=config.seed_top_n_per_category,
        )
    elif backend == "sentence_transformer":
        payload = _create_sentence_transformer_seed(
            keywords,
            model_id=config.seed_model_id,
            threshold=config.seed_similarity_threshold,
            batch_size=config.embedding_batch_size,
            top_n_per_category=config.seed_top_n_per_category,
        )
    else:
        try:
            payload = _create_sentence_transformer_seed(
                keywords,
                model_id=config.seed_model_id,
                threshold=config.seed_similarity_threshold,
                batch_size=config.embedding_batch_size,
                top_n_per_category=config.seed_top_n_per_category,
            )
        except Exception as exc:
            print(f"[seed] Embedding seed expansion failed; falling back to TF-IDF. Reason: {exc}")
            payload = _create_tfidf_seed(
                keywords,
                threshold=config.seed_similarity_threshold,
                top_n_per_category=config.seed_top_n_per_category,
            )

    write_json(config.seed_path, payload)
    print(f"[seed] seed.json -> {config.seed_path}")
    return config.seed_path
