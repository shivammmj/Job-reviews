from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

try:
    from .config import PipelineConfig
    from .utils import clean_text, detect_text_column, ensure_dir, normalize_for_nlp
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import clean_text, detect_text_column, ensure_dir, normalize_for_nlp


def _load_texts(paths: dict[str, Path]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    sources: list[str] = []
    for source_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        frame = pd.read_csv(path, low_memory=False)
        text_column = detect_text_column(frame.columns)
        for value in frame[text_column].map(clean_text):
            if value:
                texts.append(value)
                sources.append(source_name)
    return texts, sources


def _spacy_preprocess(texts: list[str], model_name: str) -> list[str]:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy preprocessing requested, but spaCy is not installed.") from exc

    nlp = spacy.load(model_name, disable=["parser"])
    allowed_pos = {"ADJ", "NOUN", "PROPN", "VERB"}
    processed: list[str] = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in allowed_pos
            and not token.is_stop
            and not token.is_punct
            and len(token.lemma_) > 2
        ]
        entities = [entity.text.lower() for entity in doc.ents if len(entity.text) > 2]
        processed.append(" ".join(tokens + entities))
    return processed


def _basic_preprocess(texts: list[str]) -> list[str]:
    return [normalize_for_nlp(text) for text in texts]


def extract_keywords(config: PipelineConfig, split_paths: dict[str, Path] | None = None) -> Path:
    """Create keywords.csv from the four review-side files."""
    ensure_dir(config.intermediate_dir)
    paths = split_paths or config.split_paths
    texts, _sources = _load_texts(paths)
    if not texts:
        raise ValueError("No review text available for keyword extraction.")

    print(f"[keywords] Building corpus from {len(texts):,} review-side rows")
    if config.use_spacy:
        processed_texts = _spacy_preprocess(texts, config.spacy_model)
        preprocessing = "spacy_pos_lemma_ner"
    else:
        processed_texts = _basic_preprocess(texts)
        preprocessing = "basic_regex_stopword"

    min_df = config.keyword_min_df
    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, config.keyword_ngram_max),
            min_df=min_df,
            max_features=config.keyword_max_features,
        )
        matrix = vectorizer.fit_transform(processed_texts)
    except ValueError:
        min_df = 1
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, config.keyword_ngram_max),
            min_df=min_df,
            max_features=config.keyword_max_features,
        )
        matrix = vectorizer.fit_transform(processed_texts)

    terms = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    output = pd.DataFrame(
        {
            "keyword": terms,
            "frequency": counts.astype(int),
            "ngram_len": [len(term.split()) for term in terms],
            "preprocessing": preprocessing,
            "min_df_used": min_df,
        }
    ).sort_values(["frequency", "keyword"], ascending=[False, True])

    output.to_csv(config.keywords_path, index=False)
    print(f"[keywords] {len(output):,} keywords -> {config.keywords_path}")
    return config.keywords_path
