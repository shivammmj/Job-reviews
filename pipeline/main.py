from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .classify_reviews import classify_review_files
    from .cluster_reviews import cluster_review_files
    from .config import PipelineConfig
    from .nlp_keyword_extraction import extract_keywords
    from .seed_expansion import create_seed_json
    from .split_reviews import split_reviews
    from .utils import write_json
except ImportError:  # pragma: no cover - supports python pipeline/main.py
    from classify_reviews import classify_review_files
    from cluster_reviews import cluster_review_files
    from config import PipelineConfig
    from nlp_keyword_extraction import extract_keywords
    from seed_expansion import create_seed_json
    from split_reviews import split_reviews
    from utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular review preprocessing and HR/Non-HR classification pipeline."
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Rows per raw file; 0 means full data.")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--intermediate-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--glassdoor-input", default=None)
    parser.add_argument("--ambitionbox-input", default=None)
    parser.add_argument(
        "--seed-backend",
        choices=["paper_embedding_expanded", "paper_exact", "auto", "tfidf", "sentence_transformer"],
        default=None,
        help="Seed creation mode. paper_embedding_expanded follows category_keyword_extension_final.py.",
    )
    parser.add_argument(
        "--classification-backend",
        choices=["auto", "tfidf", "sentence_transformer"],
        default=None,
        help="Backend for assigning HR/Non-HR labels.",
    )
    parser.add_argument("--seed-model-id", default=None)
    parser.add_argument("--spacy-model", default=None)
    parser.add_argument("--llama-model-id", default=None)
    parser.add_argument("--run-clustering", action="store_true", help="Cluster Non HR rows after classification.")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--cluster-embedding-model-id", default=None)
    parser.add_argument("--cluster-reducer", choices=["auto", "umap", "pca"], default=None)
    parser.add_argument("--cluster-algorithm", choices=["auto", "hdbscan", "dbscan", "kmeans"], default=None)
    parser.add_argument("--cluster-umap-components", type=int, default=None)
    parser.add_argument("--cluster-umap-neighbors", type=int, default=None)
    parser.add_argument("--cluster-umap-min-dist", type=float, default=None)
    parser.add_argument("--cluster-min-cluster-size", type=int, default=None)
    parser.add_argument("--cluster-min-samples", type=int, default=None)
    parser.add_argument("--cluster-random-state", type=int, default=None)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-keywords", action="store_true")
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    return parser.parse_args()


def _ensure_raw_inputs(config: PipelineConfig) -> None:
    defaults = {
        config.glassdoor_input: config.root_dir
        / "data"
        / "cleaned_reviews"
        / "glassdoor"
        / "all_glassdoor_reviews.csv",
        config.ambitionbox_input: config.root_dir
        / "data"
        / "cleaned_reviews"
        / "ambition_box"
        / "all_ambitionbox_reviews.csv",
    }
    for target, source in defaults.items():
        if target.exists():
            continue
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"[setup] copied {source} -> {target}")
        else:
            raise FileNotFoundError(
                f"Required raw input missing: {target}. Also could not find fallback source: {source}"
            )


def _write_manifest(
    config: PipelineConfig,
    split_paths: dict[str, Path],
    keywords_path: Path,
    seed_path: Path,
    classified_paths: dict[str, Path],
    clustered_paths: dict[str, Path],
) -> Path:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "split_paths": {key: str(value) for key, value in split_paths.items()},
        "keywords_path": str(keywords_path),
        "seed_path": str(seed_path),
        "classified_paths": {key: str(value) for key, value in classified_paths.items()},
        "clustered_paths": {key: str(value) for key, value in clustered_paths.items()},
    }
    manifest_path = config.report_dir / "pipeline_run_manifest.json"
    write_json(manifest_path, payload)
    print(f"[report] manifest -> {manifest_path}")
    return manifest_path


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_args(args)
    _ensure_raw_inputs(config)

    print("[pipeline] Starting review pipeline")
    print(f"[pipeline] max_rows={config.max_rows or 'full'}")
    print(f"[pipeline] seed_backend={config.seed_backend}")
    print(f"[pipeline] classification_backend={config.classification_backend}")
    print(f"[pipeline] run_clustering={config.run_clustering}")

    if args.skip_split:
        split_paths = config.split_paths
        print("[split] skipped; using existing intermediate split files")
    else:
        split_paths = split_reviews(config)

    if args.skip_keywords:
        keywords_path = config.keywords_path
        print("[keywords] skipped; using existing keywords.csv")
    else:
        keywords_path = extract_keywords(config, split_paths)

    if args.skip_seed:
        seed_path = config.seed_path
        print("[seed] skipped; using existing seed.json")
    else:
        seed_path = create_seed_json(config, keywords_path)

    if args.skip_classification:
        classified_paths = config.classified_paths
        print("[classify] skipped; using existing classified files")
    else:
        classified_paths = classify_review_files(config, split_paths, seed_path)

    if config.run_clustering:
        clustered_paths = cluster_review_files(config, classified_paths)
    else:
        clustered_paths = {}
        print("[cluster] skipped; use --run-clustering to create Non HR cluster files")

    _write_manifest(config, split_paths, keywords_path, seed_path, classified_paths, clustered_paths)
    print("[pipeline] Completed")


if __name__ == "__main__":
    main()
