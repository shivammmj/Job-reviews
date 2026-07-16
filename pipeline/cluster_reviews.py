from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

try:
    from .config import PipelineConfig
    from .utils import configure_quiet_model_loading, clean_text, detect_text_column, ensure_dir
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from utils import configure_quiet_model_loading, clean_text, detect_text_column, ensure_dir


def _load_non_hr_rows(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Classified file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    text_column = detect_text_column(frame.columns)
    if "classification" not in frame.columns:
        raise ValueError(f"Missing classification column in {path}")
    non_hr = frame[frame["classification"].astype(str).str.lower() == "non hr"].copy()
    non_hr[text_column] = non_hr[text_column].map(clean_text)
    non_hr = non_hr[non_hr[text_column].str.len() > 0].reset_index(drop=True)
    return non_hr, text_column


def _load_embedding_model(config: PipelineConfig) -> object:
    configure_quiet_model_loading()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.cluster_embedding_model_id)


def _embed_texts(texts: list[str], config: PipelineConfig, model: object) -> np.ndarray:
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=config.embedding_batch_size,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _reduce_embeddings(
    embeddings: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, str]:
    if len(embeddings) <= 2:
        return embeddings, "none_too_few_rows"

    reducer = config.cluster_reducer.lower().replace("-", "_")
    n_components = min(config.cluster_umap_components, max(2, len(embeddings) - 1))

    if reducer in {"umap", "auto"}:
        try:
            from umap import UMAP

            n_neighbors = min(config.cluster_umap_neighbors, max(2, len(embeddings) - 1))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="n_jobs value 1 overridden.*",
                    category=UserWarning,
                )
                reduced = UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=config.cluster_umap_min_dist,
                    metric="cosine",
                    random_state=config.cluster_random_state,
                ).fit_transform(embeddings)
            return np.asarray(reduced, dtype=np.float32), "umap"
        except Exception as exc:
            print(f"[cluster] UMAP reduction failed; falling back to PCA. Reason: {exc}")

    if reducer not in {"pca", "auto", "umap"}:
        print(f"[cluster] Unknown reducer={config.cluster_reducer}; falling back to PCA")

    n_components = min(n_components, embeddings.shape[1], len(embeddings) - 1)
    reduced = PCA(n_components=n_components, random_state=config.cluster_random_state).fit_transform(
        embeddings
    )
    return np.asarray(reduced, dtype=np.float32), "pca"


def _cluster_reduced(
    reduced: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, str]:
    if len(reduced) < 2:
        return np.full(len(reduced), -1, dtype=int), "none_too_few_rows"

    algorithm = config.cluster_algorithm.lower().replace("-", "_")
    min_cluster_size = max(2, min(config.cluster_min_cluster_size, len(reduced)))
    min_samples = max(1, min(config.cluster_min_samples, len(reduced)))

    if algorithm in {"hdbscan", "auto"}:
        try:
            import hdbscan

            labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                prediction_data=False,
            ).fit_predict(reduced)
            return labels.astype(int), "hdbscan"
        except Exception as exc:
            print(f"[cluster] HDBSCAN failed; falling back to DBSCAN. Reason: {exc}")

    if algorithm == "kmeans":
        n_clusters = max(2, min(8, len(reduced) // min_cluster_size))
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=config.cluster_random_state,
            n_init="auto",
        ).fit_predict(reduced)
        return labels.astype(int), "kmeans"

    labels = DBSCAN(eps=0.45, min_samples=min_samples, metric="euclidean").fit_predict(reduced)
    return labels.astype(int), "dbscan"


def _add_cluster_columns(
    frame: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    reducer_used: str,
    cluster_algorithm_used: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    output = frame.copy()
    output["cluster_id"] = labels
    output["is_cluster_noise"] = labels == -1

    counts = pd.Series(labels).value_counts().to_dict()
    output["cluster_size"] = [int(counts[label]) for label in labels]

    if reduced.shape[1] >= 2:
        output["cluster_x"] = reduced[:, 0]
        output["cluster_y"] = reduced[:, 1]
    elif reduced.shape[1] == 1:
        output["cluster_x"] = reduced[:, 0]
        output["cluster_y"] = 0.0
    else:
        output["cluster_x"] = 0.0
        output["cluster_y"] = 0.0

    output["reducer_used"] = reducer_used
    output["cluster_algorithm_used"] = cluster_algorithm_used
    output["cluster_embedding_model_id"] = config.cluster_embedding_model_id
    output["cluster_min_cluster_size"] = config.cluster_min_cluster_size
    output["cluster_min_samples"] = config.cluster_min_samples
    return output


def _write_empty_cluster_file(
    frame: pd.DataFrame,
    output_path: Path,
    reason: str,
    config: PipelineConfig,
) -> Path:
    output = frame.copy()
    output["cluster_id"] = pd.Series(dtype="int64")
    output["is_cluster_noise"] = pd.Series(dtype="bool")
    output["cluster_size"] = pd.Series(dtype="int64")
    output["cluster_x"] = pd.Series(dtype="float64")
    output["cluster_y"] = pd.Series(dtype="float64")
    output["reducer_used"] = reason
    output["cluster_algorithm_used"] = reason
    output["cluster_embedding_model_id"] = config.cluster_embedding_model_id
    output.to_csv(output_path, index=False)
    print(f"[cluster] {output_path.name}: no Non HR rows -> {output_path}")
    return output_path


def _cluster_one_file(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    embedding_model: object,
) -> Path:
    frame, text_column = _load_non_hr_rows(input_path)
    if frame.empty:
        return _write_empty_cluster_file(frame, output_path, "no_non_hr_rows", config)

    texts = frame[text_column].map(clean_text).tolist()
    print(f"[cluster] {input_path.name}: embedding {len(texts):,} Non HR reviews")
    embeddings = _embed_texts(texts, config, embedding_model)
    reduced, reducer_used = _reduce_embeddings(embeddings, config)
    labels, algorithm_used = _cluster_reduced(reduced, config)

    output = _add_cluster_columns(
        frame=frame,
        reduced=reduced,
        labels=labels,
        reducer_used=reducer_used,
        cluster_algorithm_used=algorithm_used,
        config=config,
    )
    output.to_csv(output_path, index=False)

    cluster_count = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))
    print(
        "[cluster] "
        f"{input_path.name}: {cluster_count:,} clusters, "
        f"{noise_count:,} noise rows -> {output_path}"
    )
    return output_path


def cluster_review_files(
    config: PipelineConfig,
    classified_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Cluster only Non HR rows from each classified review-side file."""
    ensure_dir(config.output_dir)
    inputs = classified_paths or config.classified_paths
    written: dict[str, Path] = {}
    embedding_model = _load_embedding_model(config)
    for name, input_path in inputs.items():
        output_path = config.clustered_paths[name]
        written[name] = _cluster_one_file(input_path, output_path, config, embedding_model)
    return written
