# Review Pipeline Dependency And Run Guide

## 1. Create The Environment

From the project root:

```bash
python3 -m venv .venv_pipeline
source .venv_pipeline/bin/activate
pip install --upgrade pip
pip install -r pipeline/requirements.txt
```

Optional UMAP/HDBSCAN dependencies for later visualization and density clustering:

```bash
pip install -r pipeline/requirements-optional.txt
```

Optional spaCy model for linguistic preprocessing:

```bash
python -m spacy download en_core_web_sm
```

## 2. Configure The Pipeline

The pipeline includes `pipeline/.env.example` as the reusable template. Copy it to either `pipeline/.env` or project-root `.env` and edit paths/settings if needed.

Important toggles:

- `SEED_BACKEND=paper_exact` writes only the exact paper-derived vocabulary to `seed.json`.
- `SEED_BACKEND=tfidf`, `sentence_transformer`, and `auto` are accepted for older command compatibility, but corpus keyword expansion is disabled in the current paper-exact pipeline.
- `CLASSIFICATION_BACKEND=tfidf` runs fast and is useful for smoke tests.
- `CLASSIFICATION_BACKEND=sentence_transformer` forces embedding-based classification.
- `CLASSIFICATION_BACKEND=auto` tries embeddings first, then falls back to TF-IDF.
- `CLASSIFICATION_THRESHOLD` is used for embedding classification.
- `TFIDF_CLASSIFICATION_THRESHOLD` is used for TF-IDF classification because TF-IDF cosine scores are usually smaller.
- `HR_CATEGORY_KEYS` controls which category scores make a review `HR`.
- `MAX_ROWS=0` means full dataset; use a small number for testing.
- `LLAMA_ENABLED=false` is reserved for the later Llama validation/labeling step.

## 3. Place Input Files

The pipeline expects:

```text
pipeline/data/raw/all_glassdoor_reviews.csv
pipeline/data/raw/all_ambitionbox_reviews.csv
```

These were copied from:

```text
data/cleaned_reviews/glassdoor/all_glassdoor_reviews.csv
data/cleaned_reviews/ambition_box/all_ambitionbox_reviews.csv
```

## 4. Smoke Test

Use TF-IDF classification first to validate the full flow without model downloads:

```bash
python pipeline/main.py --max-rows 100 --seed-backend paper_exact --classification-backend tfidf
```

## 5. Full Run

For the default full run:

```bash
python pipeline/main.py
```

For exact paper seeds with embedding-based classification:

```bash
python pipeline/main.py --seed-backend paper_exact --classification-backend sentence_transformer
```

## 6. Expected Outputs

Intermediate files:

```text
pipeline/data/intermediate/pros_gd.csv
pipeline/data/intermediate/cons_gd.csv
pipeline/data/intermediate/likes_am.csv
pipeline/data/intermediate/dislikes_am.csv
pipeline/data/intermediate/keywords.csv
pipeline/data/intermediate/seed.json
```

Classified files:

```text
pipeline/data/outputs/pros_gd_classified.csv
pipeline/data/outputs/cons_gd_classified.csv
pipeline/data/outputs/likes_am_classified.csv
pipeline/data/outputs/dislikes_am_classified.csv
```

The classified files contain the original review text, metadata, category scores, and final `classification` value (`HR` or `Non HR`).

## 7. Kaggle Notes

On Kaggle, install core dependencies in a notebook cell:

```bash
pip install -r /kaggle/working/review_pipeline/pipeline/requirements.txt
```

Install optional clustering dependencies only when you reach the UMAP/HDBSCAN stage:

```bash
pip install -r /kaggle/working/review_pipeline/pipeline/requirements-optional.txt
```

If `hdbscan` fails to build, continue without it for the first HR/non-HR pipeline run. It is only needed for the later density-clustering stage.

Use TF-IDF mode first:

```bash
python /kaggle/working/review_pipeline/pipeline/main.py --max-rows 100 --seed-backend paper_exact --classification-backend tfidf
```
