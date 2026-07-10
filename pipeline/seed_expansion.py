from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import PipelineConfig
    from .utils import clean_text, write_json
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import clean_text, write_json


BASE_SEED_CATEGORIES: dict[str, list[str]] = {
    "economic_psychological_benefits": [
        "salary",
        "pay",
        "compensation",
        "bonus",
        "benefits",
        "perks",
        "insurance",
        "health",
        "wellness",
        "work-life balance",
        "work life balance",
        "wfh",
        "remote",
        "flexible",
        "flexibility",
        "leave",
        "vacation",
        "holiday",
        "wellbeing",
        "well-being",
        "monetary",
        "hike",
        "increment",
        "appraisal",
        "stock",
        "esop",
        "retirement",
        "pension",
        "allowance",
        "reimbursement",
        "medical",
        "dental",
        "fitness",
        "gym",
        "mental health",
        "ctc",
        "package",
    ],
    "workplace_environment_culture_relationships": [
        "community",
        "co-worker",
        "coworker",
        "culture",
        "workplace",
        "environment",
        "collaboration",
        "friendly",
        "atmosphere",
        "team",
        "colleagues",
        "supportive",
        "inclusive",
        "respect",
        "diversity",
        "values",
        "relationship",
        "peer",
        "manager",
        "supervisor",
        "leader",
        "mentor",
        "welcoming",
        "positive culture",
        "work culture",
        "work environment",
        "office culture",
        "toxic",
        "politics",
        "hierarchy",
        "bureaucracy",
        "micromanagement",
    ],
    "product_belief": [
        "product",
        "products",
        "quality",
        "innovation",
        "innovative",
        "technology",
        "cutting-edge",
        "cutting edge",
        "state-of-the-art",
        "state of the art",
        "best product",
        "amazing product",
        "great product",
        "cool product",
        "craftsmanship",
        "design",
        "engineering",
        "service",
        "solution",
        "platform",
    ],
    "job_satisfaction": [
        "satisfaction",
        "satisfying",
        "satisfied",
        "meaningful",
        "gratification",
        "fulfilling",
        "rewarding",
        "enjoy",
        "enjoyable",
        "love my job",
        "love working",
        "happy",
        "pleasure",
        "great place to work",
        "best place",
        "good place",
        "amazing place",
        "wonderful place",
        "awesome place",
        "interesting work",
        "challenging work",
        "exciting",
        "fun",
    ],
    "justice": [
        "fair",
        "fairness",
        "justice",
        "just",
        "equal",
        "equality",
        "equitable",
        "transparent",
        "transparency",
        "unbiased",
        "bias",
        "biased",
        "discrimination",
        "discriminate",
        "favoritism",
        "favouritism",
        "nepotism",
        "merit",
        "meritocracy",
        "honest",
        "ethical",
        "integrity",
    ],
    "employee_brand_identification": [
        "family",
        "second home",
        "home away",
        "belong",
        "belonging",
        "proud",
        "pride",
        "identity",
        "identify",
        "part of",
        "one big family",
        "our company",
        "our team",
        "our brand",
        "my company",
        "ownership",
    ],
    "employee_brand_love": [
        "love",
        "passionate",
        "passion",
        "adore",
        "amazing company",
        "best company",
        "dream company",
        "dream job",
        "love this company",
        "love the company",
        "love working here",
        "highly recommend",
        "would recommend",
        "recommend to",
        "recommend this",
        "strongly recommend",
    ],
    "inner_self_expressive_brands": [
        "values align",
        "my values",
        "my belief",
        "believe in",
        "personal values",
        "share my",
        "reflects my",
        "my personality",
        "resonate",
        "align with",
        "purpose",
        "mission",
        "vision",
        "meaningful",
        "making a difference",
        "impact",
        "contribute",
        "personal growth",
        "self-development",
    ],
    "brand_reputation": [
        "great brand",
        "brand name",
        "brand value",
        "reputed",
        "reputation",
        "respected",
        "prestigious",
        "premium",
        "well-known",
        "well known",
        "famous",
        "renowned",
        "top company",
        "top brand",
        "best brand",
        "world-class",
        "world class",
        "global brand",
        "market leader",
        "industry leader",
        "trusted brand",
        "good brand",
        "strong brand",
        "resume",
        "cv",
    ],
    "brand_authenticity": [
        "authentic",
        "authenticity",
        "genuine",
        "transparency",
        "transparent",
        "trust",
        "trustworthy",
        "accountable",
        "accountability",
        "morals",
        "ethics",
        "ethical",
        "integrity",
        "walk the talk",
        "true to",
        "honest",
        "honesty",
        "sincere",
        "sincerity",
        "credible",
        "credibility",
    ],
    "social_self_expressive_brands": [
        "proud to tell",
        "proud to say",
        "proud to work",
        "proudly",
        "brag",
        "status",
        "prestige",
        "impressed",
        "impressive",
        "admire",
        "look up to",
        "cool factor",
        "social image",
        "tell people",
        "feel important",
        "people think",
    ],
}


PAPER_SEED_SOURCE = "reference/category_keyword_table.md / Raj & Rehman (2025)"


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
            _add_seed(category_terms, category, seed, 1.0, "paper_seed")
    return category_terms


def _create_paper_exact_seed() -> dict:
    payload = _new_seed_payload("paper_exact", PAPER_SEED_SOURCE, 1.0)
    payload["_meta"]["seed_source"] = PAPER_SEED_SOURCE
    payload["_meta"]["uses_corpus_expansion"] = False
    return _finalize_payload(
        payload,
        _base_category_terms(),
        top_n_per_category=max(len(terms) for terms in BASE_SEED_CATEGORIES.values()),
    )

def create_seed_json(config: PipelineConfig, keywords_path: Path | None = None) -> Path:
    """Create seed.json using only exact paper-derived seed terms."""
    backend = config.seed_backend.lower().replace("-", "_")
    if backend not in {"paper_exact", "auto", "tfidf", "sentence_transformer"}:
        raise ValueError(f"Unsupported seed backend: {config.seed_backend}")

    if backend != "paper_exact":
        print(
            "[seed] Corpus keyword expansion is disabled. "
            f"Received backend={backend}, but writing exact paper seeds only."
        )
    else:
        print("[seed] Writing exact paper-derived seed terms only")

    payload = _create_paper_exact_seed()
    write_json(config.seed_path, payload)
    print(f"[seed] seed.json -> {config.seed_path}")
    return config.seed_path
