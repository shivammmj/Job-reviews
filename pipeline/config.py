from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from .utils import (
        ensure_dir,
        env_bool,
        env_float,
        env_int,
        load_default_env_files,
        resolve_path,
    )
except ImportError:  # pragma: no cover - supports python pipeline/main.py
    from utils import (
        ensure_dir,
        env_bool,
        env_float,
        env_int,
        load_default_env_files,
        resolve_path,
    )


load_default_env_files()


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path
    raw_dir: Path
    intermediate_dir: Path
    output_dir: Path
    report_dir: Path
    cache_dir: Path
    glassdoor_input: Path
    ambitionbox_input: Path
    max_rows: int
    batch_size: int
    embedding_batch_size: int
    keyword_max_features: int
    keyword_min_df: int
    keyword_ngram_max: int
    seed_model_id: str
    seed_backend: str
    seed_similarity_threshold: float
    seed_top_n_per_category: int
    classification_backend: str
    classification_threshold: float
    tfidf_classification_threshold: float
    hr_category_keys: list[str]
    use_spacy: bool
    spacy_model: str
    llama_enabled: bool
    llama_model_id: str
    run_clustering: bool
    cluster_embedding_model_id: str
    cluster_reducer: str
    cluster_algorithm: str
    cluster_umap_components: int
    cluster_umap_neighbors: int
    cluster_umap_min_dist: float
    cluster_min_cluster_size: int
    cluster_min_samples: int
    cluster_random_state: int

    @classmethod
    def from_args(cls, args: object | None = None) -> "PipelineConfig":
        root_dir = Path(__file__).resolve().parent.parent

        def arg(name: str, default: object = None) -> object:
            if args is None:
                return default
            return getattr(args, name, default)

        raw_dir = resolve_path(str(arg("raw_dir") or os.getenv("PIPELINE_RAW_DIR", "pipeline/data/raw")), root_dir)
        intermediate_dir = resolve_path(
            str(
                arg("intermediate_dir")
                or os.getenv("PIPELINE_INTERMEDIATE_DIR", "pipeline/data/intermediate")
            ),
            root_dir,
        )
        output_dir = resolve_path(
            str(arg("output_dir") or os.getenv("PIPELINE_OUTPUT_DIR", "pipeline/data/outputs")),
            root_dir,
        )
        report_dir = resolve_path(
            str(arg("report_dir") or os.getenv("PIPELINE_REPORT_DIR", "pipeline/data/reports")),
            root_dir,
        )
        cache_dir = resolve_path(
            str(arg("cache_dir") or os.getenv("PIPELINE_CACHE_DIR", "pipeline/data/cache")),
            root_dir,
        )

        glassdoor_input = resolve_path(
            str(
                arg("glassdoor_input")
                or os.getenv("GLASSDOOR_INPUT", "pipeline/data/raw/all_glassdoor_reviews.csv")
            ),
            root_dir,
        )
        ambitionbox_input = resolve_path(
            str(
                arg("ambitionbox_input")
                or os.getenv("AMBITIONBOX_INPUT", "pipeline/data/raw/all_ambitionbox_reviews.csv")
            ),
            root_dir,
        )

        max_rows_arg = arg("max_rows")
        seed_backend_arg = arg("seed_backend")
        classification_backend_arg = arg("classification_backend")
        run_clustering_arg = arg("run_clustering")
        skip_clustering_arg = arg("skip_clustering")
        run_clustering = env_bool("RUN_CLUSTERING", False)
        if run_clustering_arg:
            run_clustering = True
        if skip_clustering_arg:
            run_clustering = False

        config = cls(
            root_dir=root_dir,
            raw_dir=raw_dir,
            intermediate_dir=intermediate_dir,
            output_dir=output_dir,
            report_dir=report_dir,
            cache_dir=cache_dir,
            glassdoor_input=glassdoor_input,
            ambitionbox_input=ambitionbox_input,
            max_rows=int(max_rows_arg) if max_rows_arg is not None else env_int("MAX_ROWS", 0),
            batch_size=env_int("BATCH_SIZE", 10000),
            embedding_batch_size=env_int("EMBEDDING_BATCH_SIZE", 64),
            keyword_max_features=env_int("KEYWORD_MAX_FEATURES", 2500),
            keyword_min_df=env_int("KEYWORD_MIN_DF", 2),
            keyword_ngram_max=env_int("KEYWORD_NGRAM_MAX", 3),
            seed_model_id=str(
                arg("seed_model_id")
                or os.getenv("SEED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            seed_backend=str(seed_backend_arg or os.getenv("SEED_BACKEND", "paper_embedding_expanded")),
            seed_similarity_threshold=env_float("SEED_SIMILARITY_THRESHOLD", 0.60),
            seed_top_n_per_category=env_int("SEED_TOP_N_PER_CATEGORY", 250),
            classification_backend=str(
                classification_backend_arg or os.getenv("CLASSIFICATION_BACKEND", "auto")
            ),
            classification_threshold=env_float("CLASSIFICATION_THRESHOLD", 0.50),
            tfidf_classification_threshold=env_float("TFIDF_CLASSIFICATION_THRESHOLD", 0.10),
            hr_category_keys=[
                value.strip()
                for value in os.getenv(
                    "HR_CATEGORY_KEYS",
                    "economic_psychological_benefits,workplace_environment_culture_relationships,"
                    "job_satisfaction,justice,employee_brand_identification",
                ).split(",")
                if value.strip()
            ],
            use_spacy=env_bool("USE_SPACY", False),
            spacy_model=str(arg("spacy_model") or os.getenv("SPACY_MODEL", "en_core_web_sm")),
            llama_enabled=env_bool("LLAMA_ENABLED", False),
            llama_model_id=str(
                arg("llama_model_id")
                or os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
            ),
            run_clustering=run_clustering,
            cluster_embedding_model_id=str(
                arg("cluster_embedding_model_id")
                or os.getenv("CLUSTER_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            cluster_reducer=str(arg("cluster_reducer") or os.getenv("CLUSTER_REDUCER", "umap")),
            cluster_algorithm=str(arg("cluster_algorithm") or os.getenv("CLUSTER_ALGORITHM", "hdbscan")),
            cluster_umap_components=int(
                arg("cluster_umap_components") or env_int("CLUSTER_UMAP_COMPONENTS", 10)
            ),
            cluster_umap_neighbors=int(
                arg("cluster_umap_neighbors") or env_int("CLUSTER_UMAP_NEIGHBORS", 15)
            ),
            cluster_umap_min_dist=float(
                arg("cluster_umap_min_dist") or env_float("CLUSTER_UMAP_MIN_DIST", 0.0)
            ),
            cluster_min_cluster_size=int(
                arg("cluster_min_cluster_size") or env_int("CLUSTER_MIN_CLUSTER_SIZE", 15)
            ),
            cluster_min_samples=int(
                arg("cluster_min_samples") or env_int("CLUSTER_MIN_SAMPLES", 5)
            ),
            cluster_random_state=int(
                arg("cluster_random_state") or env_int("CLUSTER_RANDOM_STATE", 42)
            ),
        )

        for directory in [
            config.raw_dir,
            config.intermediate_dir,
            config.output_dir,
            config.report_dir,
            config.cache_dir,
        ]:
            ensure_dir(directory)

        return config

    @property
    def split_paths(self) -> dict[str, Path]:
        return {
            "pros_gd": self.intermediate_dir / "pros_gd.csv",
            "cons_gd": self.intermediate_dir / "cons_gd.csv",
            "likes_am": self.intermediate_dir / "likes_am.csv",
            "dislikes_am": self.intermediate_dir / "dislikes_am.csv",
        }

    @property
    def keywords_path(self) -> Path:
        return self.intermediate_dir / "keywords.csv"

    @property
    def seed_path(self) -> Path:
        return self.intermediate_dir / "seed.json"

    @property
    def classified_paths(self) -> dict[str, Path]:
        return {
            "pros_gd": self.output_dir / "pros_gd_classified.csv",
            "cons_gd": self.output_dir / "cons_gd_classified.csv",
            "likes_am": self.output_dir / "likes_am_classified.csv",
            "dislikes_am": self.output_dir / "dislikes_am_classified.csv",
        }

    @property
    def clustered_paths(self) -> dict[str, Path]:
        return {
            "pros_gd": self.output_dir / "pros_gd_non_hr_clustered.csv",
            "cons_gd": self.output_dir / "cons_gd_non_hr_clustered.csv",
            "likes_am": self.output_dir / "likes_am_non_hr_clustered.csv",
            "dislikes_am": self.output_dir / "dislikes_am_non_hr_clustered.csv",
        }
