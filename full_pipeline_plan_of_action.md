# Full Pipeline Plan Of Action

## Goal

Build a modular review clustering pipeline that starts from the two cleaned consolidated platform files:

```text
data/cleaned_reviews/glassdoor/all_glassdoor_reviews.csv
data/cleaned_reviews/ambition_box/all_ambitionbox_reviews.csv
```

The `_tagged.csv` files are not the main input. They can be used only for checking, debugging, or comparison.

The final pipeline should produce clustered review-level outputs for:

```text
pros_gd
cons_gd
likes_am
dislikes_am
```

Each final review row should have:

```text
original review text
platform
source field
positive-side / negative-side label
HR / Non-HR classification
advocacy factor scores
cluster number
cluster size
dominant cluster theme
```


## Global Toggle Design

The pipeline should be configurable so we can switch techniques without rewriting code.

```yaml
model_mode: baseline

models:
  sentence_transformer:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 256

  llama:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    torch_dtype: bfloat16
    device_map: auto
    batch_size: 8
    max_length: 256

nlp_enrichment:
  enabled: true
  engine: spacy
  use_lemmas: true
  use_pos_tags: true
  use_noun_chunks: true
  use_ner: false
  use_nltk_collocations: false
```

Recommended default:

```text
pandas + regex + spaCy + TF-IDF + SVD + seed similarity + numeric factors + MiniBatchKMeans + NetworkX Louvain
```

Optional comparison or validation:

```text
sentence-transformers
Llama embeddings
Llama validation
UMAP visualization
HDBSCAN exploratory clustering
```


# Step 1: Read The Two Cleaned Consolidated Platform Files

## What We Do

Load the two main cleaned review datasets into Python:

```text
data/cleaned_reviews/glassdoor/all_glassdoor_reviews.csv
data/cleaned_reviews/ambition_box/all_ambitionbox_reviews.csv
```

We do not use the `_tagged.csv` files as the main pipeline input.

## Technique Used

Use `pandas.read_csv()`.

```python
import pandas as pd

glassdoor_df = pd.read_csv(
    "data/cleaned_reviews/glassdoor/all_glassdoor_reviews.csv",
    low_memory=False,
)

ambitionbox_df = pd.read_csv(
    "data/cleaned_reviews/ambition_box/all_ambitionbox_reviews.csv",
    low_memory=False,
)
```

For large files, use chunking:

```python
glassdoor_chunks = pd.read_csv(
    "data/cleaned_reviews/glassdoor/all_glassdoor_reviews.csv",
    chunksize=100000,
    low_memory=False,
)
```

## Why

- `pandas` is reliable for CSV review data.
- The current project is already CSV-based.
- DataFrames make filtering, splitting, cleaning, grouping, and exporting straightforward.

## Output

```text
glassdoor_df
ambitionbox_df
```


# Step 2: Standardize The Columns From Both Platforms

## What We Do

Rename platform-specific columns into one common structure.

| Meaning | Glassdoor Column | AmbitionBox Column | Standard Column |
|---|---|---|---|
| Company | `Company` | `Company` | `company` |
| Positive-side text | `Pros` | `Likes` | `positive_text` |
| Negative-side text | `Cons` | `Dislikes` | `negative_text` |
| Job / role | `Job` | `Job_Profile` | `job_title` |
| Rating | `Rating` | `Overall_Rating` | `rating` |
| Date | `Date` | `Date` | `date` |

## Technique Used

Use `pandas.rename()` and column selection.

```python
glassdoor_standard = glassdoor_df.rename(columns={
    "Company": "company",
    "Pros": "positive_text",
    "Cons": "negative_text",
    "Job": "job_title",
    "Rating": "rating",
    "Date": "date",
})
glassdoor_standard["platform"] = "glassdoor"

ambitionbox_standard = ambitionbox_df.rename(columns={
    "Company": "company",
    "Likes": "positive_text",
    "Dislikes": "negative_text",
    "Job_Profile": "job_title",
    "Overall_Rating": "rating",
    "Date": "date",
})
ambitionbox_standard["platform"] = "ambitionbox"
```

Keep common columns:

```python
common_columns = [
    "platform",
    "company",
    "positive_text",
    "negative_text",
    "job_title",
    "rating",
    "date",
]

glassdoor_standard = glassdoor_standard[common_columns]
ambitionbox_standard = ambitionbox_standard[common_columns]
```

## Why

- Glassdoor and AmbitionBox use different schemas.
- Later functions should not need platform-specific column logic.
- A common schema makes the pipeline reusable and modular.

## Output

```text
glassdoor_standard
ambitionbox_standard
```


# Step 3: Split Platform Review Fields Into Positive-Side And Negative-Side Segments

## What We Do

Split each platform row into separate review-text rows.

| Platform | Source Field | Segment Type |
|---|---|---|
| Glassdoor | `Pros` | positive-side |
| Glassdoor | `Cons` | negative-side |
| AmbitionBox | `Likes` | positive-side |
| AmbitionBox | `Dislikes` | negative-side |

This is not AI sentiment detection. It is a source-side label based on where the reviewer typed the text.

Example:

```text
Pros: Good salary and flexible work
Cons: Poor management and slow promotion
```

After splitting:

```text
positive-side row -> Good salary and flexible work
negative-side row -> Poor management and slow promotion
```

## Technique Used

Use `pandas` copying and `pd.concat()`.

```python
def split_review_segments(df):
    positive = df.copy()
    positive["sentiment_side"] = "positive"
    positive["review_text"] = positive["positive_text"]

    negative = df.copy()
    negative["sentiment_side"] = "negative"
    negative["review_text"] = negative["negative_text"]

    return pd.concat([positive, negative], ignore_index=True)

glassdoor_segments = split_review_segments(glassdoor_standard)
ambitionbox_segments = split_review_segments(ambitionbox_standard)

review_segments = pd.concat(
    [glassdoor_segments, ambitionbox_segments],
    ignore_index=True,
)
```

Add source field:

```python
review_segments["source_field"] = review_segments.apply(
    lambda row: (
        "Pros" if row["platform"] == "glassdoor" and row["sentiment_side"] == "positive"
        else "Cons" if row["platform"] == "glassdoor"
        else "Likes" if row["platform"] == "ambitionbox" and row["sentiment_side"] == "positive"
        else "Dislikes"
    ),
    axis=1,
)
```

## Where NLP / LLM Can Be Used

| Option | Technique | Why |
|---|---|---|
| Source-side split only | `pandas` split by source field | Recommended default; fast and reliable |
| spaCy / NLTK quality check | token count, POS count, noun chunks | Flags low-information text |
| spaCy NER | ORG / GPE / PRODUCT extraction | Optional metadata enrichment |
| Llama validation | generative check on suspicious rows | Finds cases where source-side and wording conflict |

## Toggle

```yaml
review_splitting:
  source_side_split:
    enabled: true

  spacy_quality_check:
    enabled: true
    token_count: true
    noun_chunk_count: true

  spacy_ner:
    enabled: false

  llama_source_side_audit:
    enabled: false
    run_on: suspicious_rows_only
    model_id: meta-llama/Llama-3.1-8B-Instruct
```

## Output

```text
review_segments
```

Important columns:

```text
platform
company
source_field
sentiment_side
review_text
job_title
rating
date
```


# Step 4: Clean, Normalize, And Enrich Review Text

## What We Do

Create multiple text versions:

```text
review_text_original
review_text_clean
review_text_normalized
review_text_lemma
```

## Techniques Used

| Technique | Library / Method | Why |
|---|---|---|
| Basic cleaning | Python string methods + `re.sub()` | Remove extra whitespace |
| Normalization | lowercase + whitespace collapse | Stable deduplication |
| Lemmatization | spaCy lemmatizer | Match word variants |
| POS tagging | spaCy POS tags | Detect useful content |
| Noun chunks | spaCy noun chunks | Extract meaningful phrases |
| Collocations | NLTK bigrams/trigrams | Find repeated phrases |
| NER | spaCy NER | Optional metadata enrichment |
| Llama audit | Hugging Face Transformers | Optional suspicious-row review |

## How

```python
import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_text(text):
    return clean_text(text).lower()
```

spaCy enrichment:

```python
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_space and not token.is_punct
    )

def extract_noun_chunks(text):
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]
```

## Toggle

```yaml
text_processing:
  basic_cleaning:
    enabled: true

  normalization:
    enabled: true
    lowercase: true
    collapse_whitespace: true

  spacy_enrichment:
    enabled: true
    model: en_core_web_sm
    use_lemmas: true
    use_pos_counts: true
    use_noun_chunks: true
    use_ner: false

  nltk_collocations:
    enabled: false
    use_bigrams: true
    use_trigrams: true
    min_frequency: 3

  llama_text_audit:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    run_on: suspicious_rows_only
    max_rows: 500
```

## Recommended Default

```text
regex cleaning
+ lowercase normalization
+ spaCy lemmatization
+ spaCy POS counts
+ spaCy noun chunks
```

## Output

```text
review_segments_enriched
```

Important columns:

```text
review_text_original
review_text_clean
review_text_normalized
review_text_lemma
token_count
noun_count
verb_count
adj_count
noun_chunks
entities_optional
segment_quality
```


# Step 5: Generate Advocacy Factor Scores

## What We Do

Convert each review into numeric scores for the 11 employee advocacy factors:

```text
workplace_environment_culture_relationships
economic_psychological_benefits
job_satisfaction
justice
employee_brand_identification
brand_reputation
product_belief
employee_brand_love
inner_self_expressive_brands
brand_authenticity
social_self_expressive_brands
```

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| Regex keyword scoring | `re.compile()` keyword matching | Fast, explainable baseline |
| TF-IDF factor similarity | `TfidfVectorizer` + cosine similarity | More flexible lexical matching |
| spaCy enrichment | lemmas + noun chunks | Better phrase matching |
| NLTK collocations | bigram/trigram phrases | Expands factor dictionaries |
| Sentence-transformer | embeddings + cosine similarity | Semantic matching |
| Llama factor scoring | JSON scoring prompt | Validation or small samples |

## Recommended Default

```text
Hybrid Regex + TF-IDF + spaCy noun chunks
```

Example formula:

```text
final_factor_score =
0.60 * normalized_regex_score
+ 0.40 * tfidf_similarity_score
```

Optional stronger mode:

```text
final_factor_score =
0.40 * regex_score
+ 0.30 * tfidf_similarity_score
+ 0.30 * sentence_transformer_similarity_score
```

## Toggle

```yaml
advocacy_scoring:
  strategy: hybrid_regex_tfidf

  regex:
    enabled: true
    normalize: true
    weight: 0.60

  tfidf_similarity:
    enabled: true
    ngram_range: [1, 2]
    weight: 0.40

  spacy_enrichment:
    enabled: true
    use_lemmas: true
    use_noun_chunks: true
    noun_chunk_weight: 2.0

  nltk_collocations:
    enabled: false
    use_bigrams: true
    use_trigrams: true
    min_frequency: 3

  sentence_transformer:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 256
    weight: 0.30

  llama_factor_scoring:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    run_on: sample_or_low_confidence
    max_rows: 500
```

## Output

```text
review_segments_scored
```


# Step 6: Classify Reviews As HR Or Non-HR

## What We Do

Assign every review:

```text
HR
Non HR
```

Only `Non HR` rows go into final clustering.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| Existing column | `pandas` column lookup | Fastest if already present |
| Regex rules | HR keyword patterns with `re.compile()` | Explainable fallback |
| spaCy / NLTK rules | lemmas, noun chunks, POS tags | More precise rule scoring |
| TF-IDF classifier | class profiles + cosine similarity | Fast lexical classifier |
| Sentence-transformer | class embeddings + cosine similarity | Better semantic classifier |
| Llama classifier | JSON classification prompt | Borderline rows or validation |

## Example HR Signals

```text
human resources
recruitment
hiring
interview
onboarding
payroll
notice period
resignation
termination
layoff
HR policy
```

## Toggle

```yaml
classification:
  strategy: hybrid_rules_tfidf

  existing_column:
    enabled: true
    column_name: Classification
    use_if_present: true

  regex_rules:
    enabled: true
    weight: 0.35

  spacy_nlp_rules:
    enabled: true
    use_lemmas: true
    use_noun_chunks: true
    noun_chunk_weight: 2.0
    weight: 0.25

  tfidf_similarity:
    enabled: true
    ngram_range: [1, 2]
    weight: 0.40

  sentence_transformer:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 256
    weight: 0.50

  llama_classifier:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    run_on: borderline_rows
    max_rows: 500
    confidence_min: 0.40
    confidence_max: 0.70
```

## Recommended Default

```text
Use existing Classification if available.
If missing, use hybrid Regex + spaCy NLP rules + TF-IDF similarity.
Use Llama only for borderline rows or validation samples.
```

## Output

```text
review_segments_classified
```

Important columns:

```text
Classification
classification_confidence
classification_method
classification_reason
```


# Step 7: Filter Reviews For Clustering

## What We Do

Keep only:

```text
Classification = Non HR
```

## Technique Used

`pandas` boolean filtering:

```python
non_hr_reviews = review_segments_classified[
    review_segments_classified["Classification"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .eq("non hr")
].copy()
```

## Optional Techniques

| Option | Technique | Why |
|---|---|---|
| Exact filter | `Classification == Non HR` | Production default |
| Confidence filter | `classification_confidence >= threshold` | Keep only high-confidence rows |
| Save HR rows | `to_csv()` excluded rows | Preserve excluded reviews |
| Borderline queue | confidence range filter | Manual or Llama review |
| Llama review | prompt borderline rows | Improve uncertain labels |

## Toggle

```yaml
filtering:
  keep_classification: Non HR

  exact_filter:
    enabled: true

  confidence_filter:
    enabled: false
    min_confidence: 0.70

  save_excluded_hr:
    enabled: true
    output_path: results/hr_reviews_excluded.csv

  borderline_queue:
    enabled: true
    confidence_min: 0.40
    confidence_max: 0.70

  llama_borderline_review:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    max_rows: 500
```

## Output

```text
non_hr_reviews
```


# Step 8: Create The Four Clustering Input Groups

## What We Do

Split `Non HR` reviews into:

```text
pros_gd       -> Glassdoor Pros
cons_gd       -> Glassdoor Cons
likes_am      -> AmbitionBox Likes
dislikes_am   -> AmbitionBox Dislikes
```

## Technique Used

Use `pandas` filtering:

```python
pros_gd = non_hr_reviews[
    (non_hr_reviews["platform"] == "glassdoor")
    & (non_hr_reviews["source_field"] == "Pros")
].copy()

cons_gd = non_hr_reviews[
    (non_hr_reviews["platform"] == "glassdoor")
    & (non_hr_reviews["source_field"] == "Cons")
].copy()

likes_am = non_hr_reviews[
    (non_hr_reviews["platform"] == "ambitionbox")
    & (non_hr_reviews["source_field"] == "Likes")
].copy()

dislikes_am = non_hr_reviews[
    (non_hr_reviews["platform"] == "ambitionbox")
    & (non_hr_reviews["source_field"] == "Dislikes")
].copy()
```

## Optional Grouping Modes

| Mode | Technique | Why |
|---|---|---|
| Four source groups | platform + source field filters | Recommended default |
| Sentiment-side groups | positive_all / negative_all | Cross-platform comparison |
| Full unified group | all Non-HR reviews together | One global taxonomy |

## Toggle

```yaml
clustering_groups:
  default_mode: four_source_groups

  four_source_groups:
    enabled: true
    groups:
      pros_gd:
        platform: glassdoor
        source_field: Pros
      cons_gd:
        platform: glassdoor
        source_field: Cons
      likes_am:
        platform: ambitionbox
        source_field: Likes
      dislikes_am:
        platform: ambitionbox
        source_field: Dislikes

  sentiment_side_groups:
    enabled: false

  full_unified_group:
    enabled: false
```

## Output

```python
{
    "pros_gd": pros_gd,
    "cons_gd": cons_gd,
    "likes_am": likes_am,
    "dislikes_am": dislikes_am,
}
```


# Step 9: Parse Seed Themes For Soft Guidance

## What We Do

Load seed themes from:

```text
results/vocab_clustering_approach_and_results.md
```

Seeds guide clustering but do not force the final number of clusters.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| Markdown parser | Python regex headings and backtick terms | Current stable method |
| spaCy cleanup | lemmas and noun chunks | Normalize seed terms |
| NLTK expansion | collocations | Add real corpus phrases |
| Sentence-transformer expansion | semantic phrase similarity | Broaden seed coverage |
| Llama seed review | JSON accepted/rejected terms | Validate expanded seeds |

## Regex Parsing

```python
CLUSTER_HEADER_RE = re.compile(
    r"^### Cluster (\d+):\s*(.+)$",
    re.MULTILINE,
)

BACKTICK_RE = re.compile(r"`([^`]+)`")
```

## Toggle

```yaml
seed_themes:
  source_file: results/vocab_clustering_approach_and_results.md

  markdown_regex_parser:
    enabled: true

  spacy_cleanup:
    enabled: true
    use_lemmas: true
    use_noun_chunks: true

  nltk_collocation_expansion:
    enabled: false
    min_frequency: 5

  sentence_transformer_expansion:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2
    similarity_threshold: 0.65

  llama_seed_review:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
```

## Output

```text
seed_clusters
```


# Step 10: Build Text Features

## What We Do

Convert review text into machine-readable vectors.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| TF-IDF | `TfidfVectorizer` | Fast, scalable, explainable |
| Keep negation | custom stop words preserving `no`, `not`, `nor` | Preserves complaint meaning |
| Lemma TF-IDF | TF-IDF on `review_text_lemma` | Reduces word-form variation |
| Noun chunk TF-IDF | TF-IDF on noun chunks | Better phrase features |
| Sentence-transformer | embeddings | Better semantic representation |
| Llama embeddings | hidden-state mean pooling | Stronger GPU comparison mode |

## Default TF-IDF

```python
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    max_df=0.92,
    min_df=5,
    sublinear_tf=True,
    strip_accents="unicode",
)

tfidf_matrix = vectorizer.fit_transform(review_texts)
```

## Toggle

```yaml
text_features:
  default_backend: tfidf_svd

  tfidf:
    enabled: true
    max_features: 15000
    ngram_range: [1, 2]
    max_df: 0.92
    min_df: auto
    keep_negation_words: true
    source_text_column: review_text_clean

  spacy_lemma_tfidf:
    enabled: true

  noun_chunk_tfidf:
    enabled: false

  sentence_transformer:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2
    cache_embeddings: true

  llama_embeddings:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    torch_dtype: bfloat16
    device_map: auto
    cache_embeddings: true
```

## Output

```text
tfidf_matrix
feature_names
optional_sentence_embeddings
optional_llama_embeddings
```


# Step 11: Reduce Text Feature Dimensions

## What We Do

Reduce sparse TF-IDF features into dense semantic vectors.

## Techniques Used

| Technique | Library | Why |
|---|---|---|
| Truncated SVD | `sklearn.decomposition.TruncatedSVD` | Dense semantic features |
| Standard scaling | `StandardScaler` | Comparable dimensions |
| L2 normalization | `normalize` | Cosine-friendly vectors |
| UMAP | `umap-learn` | Visualization |
| UMAP for HDBSCAN | 15D UMAP | Optional exploratory clustering |

## Default

```python
svd = TruncatedSVD(
    n_components=96,
    random_state=42,
)

semantic_dense = svd.fit_transform(tfidf_matrix)
```

## Toggle

```yaml
dimension_reduction:
  default_method: truncated_svd

  truncated_svd:
    enabled: true
    n_components: 96

  scaling:
    enabled: true
    method: standard_scaler

  normalization:
    enabled: true
    method: l2

  umap_visualization:
    enabled: true
    n_components: 2

  umap_for_hdbscan:
    enabled: false
    n_components: 15
```

## Output

```text
semantic_dense
semantic_scaled
semantic_normalized
optional_umap_2d
optional_umap_15d
```


# Step 12: Build Seed Theme Similarity Features

## What We Do

Measure how close every review is to each seed theme.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| TF-IDF cosine | `cosine_similarity(review_tfidf, seed_tfidf)` | Default, fast |
| spaCy phrase overlap | lemmas + noun chunks | Explainable phrase matches |
| NLTK collocation overlap | bigrams/trigrams | Corpus phrase matching |
| Sentence-transformer | embedding cosine | Semantic seed matching |
| Llama seed reasoning | prompt top seed themes | Validation/UI only |

## Default

```python
seed_vectors = vectorizer.transform([
    seed.seed_text
    for seed in seed_clusters
])

seed_similarity = cosine_similarity(
    tfidf_matrix,
    seed_vectors,
)
```

## Toggle

```yaml
seed_similarity:
  default_method: tfidf_cosine

  tfidf_cosine:
    enabled: true

  spacy_phrase_overlap:
    enabled: true
    use_lemmas: true
    use_noun_chunks: true

  nltk_collocation_overlap:
    enabled: false

  sentence_transformer:
    enabled: false
    model_id: sentence-transformers/all-MiniLM-L6-v2

  llama_seed_reasoning:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
```

## Output

```text
seed_similarity_vector
top_seed_theme
top_seed_score
seed_phrase_matches_optional
```


# Step 13: Build Numeric Feature Block From Advocacy Factor Scores

## What We Do

Convert the 11 advocacy factor scores into numeric clustering features.

## Techniques Used

| Technique | Library | Why |
|---|---|---|
| Numeric column selection | `pandas` | Use only intended factor columns |
| Missing value handling | `fillna`, `np.nan_to_num` | Prevent errors |
| Standard scaling | `StandardScaler` | Comparable factor ranges |
| L2 normalization | `normalize` | Cosine-friendly numeric block |
| PCA/SVD | `PCA` or `TruncatedSVD` | Optional if many numeric columns |
| Llama explanation | prompt | UI trace only |

## Default

```python
numeric_matrix = (
    review_group[metric_columns]
    .fillna(0.0)
    .astype("float32")
    .to_numpy()
)

metric_scaled = StandardScaler().fit_transform(numeric_matrix)
metric_normalized = normalize(metric_scaled, norm="l2")
```

## Toggle

```yaml
numeric_features:
  enabled: true
  source: advocacy_factor_scores

  missing_values:
    fill_value: 0.0

  scaling:
    enabled: true
    method: standard_scaler

  normalization:
    enabled: true
    method: l2

  dimensionality_reduction:
    enabled: false
    method: pca

  llama_numeric_explanation:
    enabled: false
    run_on: ui_trace_only
```

## Output

```text
metric_scaled
metric_normalized
metric_columns_used
```


# Step 14: Combine All Feature Blocks Into One Clustering Matrix

## What We Do

Combine:

```text
semantic text features
seed similarity features
numeric advocacy factor features
```

## Technique Used

Weighted feature concatenation with `numpy.hstack`.

```python
combined_features = np.hstack([
    semantic_weight * semantic_scaled,
    seed_weight * seed_scaled,
    metric_weight * metric_scaled,
]).astype("float32")
```

Default weights:

```text
semantic_weight = 0.60
seed_weight     = 0.25
metric_weight   = 0.15
```

## Optional Approaches

| Option | Why |
|---|---|
| Feature ablation | Prove which feature blocks help |
| Sentence-transformer replacement | Stronger semantic comparison |
| Llama embedding replacement | GPU-based deeper semantic comparison |

## Toggle

```yaml
feature_combination:
  strategy: weighted_concat

  semantic_features:
    enabled: true
    source: tfidf_svd
    weight: 0.60

  seed_features:
    enabled: true
    weight: 0.25

  numeric_features:
    enabled: true
    weight: 0.15

  sentence_transformer_embeddings:
    enabled: false
    replace_semantic_features: false
    weight: 0.70

  llama_embeddings:
    enabled: false
    replace_semantic_features: false
    weight: 0.70

  ablation_runs:
    enabled: false
```

## Output

```text
combined_features
```


# Step 15: Deduplicate Reviews Before Clustering

## What We Do

Cluster each unique normalized review once, then map the cluster back to all original rows.

## Techniques Used

| Technique | Library / Method | Why |
|---|---|---|
| Exact normalized deduplication | `pandas.groupby()` | Safe and reproducible |
| Metadata preservation | aggregation rules | Keep traceability |
| Duplicate weighting | `_review_count` | Preserve frequency information |
| Near-duplicate detection | TF-IDF / MinHash / embeddings | Optional experiment |
| Low-info flag | regex + token counts + noun chunks | Explain generic clusters |

## Default

```python
unique_reviews = (
    review_group
    .groupby("review_text_normalized", sort=False)
    .agg(
        review_text_original=("review_text_original", "first"),
        _review_count=("review_text_normalized", "size"),
        **{
            column: (column, "mean")
            for column in advocacy_factor_columns
        }
    )
    .reset_index()
)
```

## Toggle

```yaml
deduplication:
  exact_normalized:
    enabled: true
    key: review_text_normalized

  preserve_original_rows:
    enabled: true

  use_duplicate_weighting:
    enabled: true
    weight_column: _review_count

  near_duplicate_detection:
    enabled: false
    method: tfidf_cosine
    similarity_threshold: 0.98

  sentence_transformer_near_duplicates:
    enabled: false

  llama_duplicate_review:
    enabled: false

  low_information_flag:
    enabled: true
```

## Output

```text
unique_reviews
original_review_rows
```


# Step 16: Create Micro-Clusters

## What We Do

Create small initial clusters from unique reviews.

These are not final clusters.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| MiniBatchKMeans | `sklearn.cluster.MiniBatchKMeans` | Production default, scalable |
| Auto micro-cluster count | sqrt heuristic capped at 384 | Adapts to file size |
| HDBSCAN | `hdbscan.HDBSCAN` | Optional exploratory clustering |
| UMAP + HDBSCAN | `umap-learn` + HDBSCAN | Optional comparison |
| Llama review | prompt representative reviews | Optional validation |

## Default

```python
kmeans = MiniBatchKMeans(
    n_clusters=requested_microclusters,
    batch_size=4096,
    n_init=10,
    random_state=42,
    reassignment_ratio=0.01,
)

microcluster_labels = kmeans.fit_predict(combined_features)
```

Micro-cluster count:

```python
heuristic = int(math.sqrt(max(n_rows, 2) / 2.0))
heuristic = max(12, heuristic)
heuristic = min(384, heuristic)
```

## Toggle

```yaml
microclustering:
  default_method: minibatch_kmeans

  minibatch_kmeans:
    enabled: true
    microcluster_cap: 384
    batch_size: 4096
    n_init: 10
    reassignment_ratio: 0.01
    random_state: 42

  hdbscan:
    enabled: false

  umap_hdbscan:
    enabled: false

  llama_microcluster_review:
    enabled: false
```

## Output

```text
microcluster_id
microcluster_count
microcluster_sizes
```


# Step 17: Build A Similarity Graph Between Micro-Clusters

## What We Do

Create a graph:

```text
node = micro-cluster
edge = similarity between micro-clusters
edge weight = similarity strength
```

## Techniques Used

| Technique | Library / Method | Why |
|---|---|---|
| Weighted micro-cluster profile | `numpy.average()` with `_review_count` | Representative micro-cluster vectors |
| Cosine similarity | `cosine_similarity` | Compare micro-clusters |
| Similarity blending | weighted `numpy` calculation | Combine semantic, seed, metric signals |
| Graph creation | `networkx.Graph` | Input for Louvain |
| Nearest-neighbor edges | `np.argsort()` | Avoid noisy full graph |
| Stability boost | `math.log1p()` | Strengthen large stable connections |

## Default

```python
combined_sim = (
    semantic_weight * semantic_sim
    + seed_weight * seed_sim
    + metric_weight * metric_sim
) / (semantic_weight + seed_weight + metric_weight)
```

Build graph:

```python
graph = nx.Graph()
graph.add_node(node_id, review_count=int(size))
graph.add_edge(node_id, neighbor_id, weight=edge_weight)
```

## Toggle

```yaml
microcluster_graph:
  similarity_weights:
    semantic: 0.60
    seed: 0.25
    metric: 0.15

  graph_neighbors: 8
  min_edge_weight: 0.30

  stability_boost:
    enabled: true

  graph_pruning:
    enabled: false

  llama_edge_review:
    enabled: false
```

## Output

```text
microcluster_graph
combined_similarity_matrix
```


# Step 18: Merge Micro-Clusters Into Final Clusters Using Louvain

## What We Do

Use Louvain community detection to merge related micro-clusters into final communities.

## Techniques Used

| Approach | Technique | Why |
|---|---|---|
| Louvain | `networkx.algorithms.community.louvain_communities` | Default organic graph clustering |
| Weighted edges | graph edge weights | Similar nodes merge more strongly |
| Resolution parameter | Louvain resolution | Controls coarse vs fine clusters |
| HDBSCAN | density clustering | Optional comparison |
| UMAP + HDBSCAN | reduced embeddings + density clustering | Optional exploratory mode |
| Llama coherence review | prompt final cluster examples | Optional validation |

## Default

```python
communities = nx.algorithms.community.louvain_communities(
    graph,
    weight="weight",
    seed=42,
    resolution=resolution,
)
```

## Toggle

```yaml
final_clustering:
  default_method: louvain

  louvain:
    enabled: true
    library: networkx
    weight_column: weight
    random_state: 42
    resolution_values: [0.8, 1.0, 1.2, 1.4, 1.6]

  hdbscan:
    enabled: false

  umap_hdbscan:
    enabled: false

  llama_cluster_coherence:
    enabled: false
```

## Output

```text
louvain_communities
```


# Step 19: Select The Best Louvain Resolution

## What We Do

Run Louvain across multiple resolution values and select the best one.

## Techniques Used

| Technique | Library / Method | Why |
|---|---|---|
| Resolution sweep | loop over `[0.8, 1.0, 1.2, 1.4, 1.6]` | Avoid manual guessing |
| Modularity | `networkx.algorithms.community.quality.modularity` | Measures community strength |
| Largest cluster penalty | arithmetic | Avoid one huge cluster |
| Model selection table | `pandas.to_csv()` | Auditable selection |
| HDBSCAN sweep | parameter grid | Optional comparison |
| Llama review | candidate summaries | Optional interpretability review |

## Formula

```text
selection score = modularity - 0.20 x largest cluster share
```

## Toggle

```yaml
resolution_selection:
  resolution_values: [0.8, 1.0, 1.2, 1.4, 1.6]
  largest_share_penalty: 0.20

  score:
    method: modularity_minus_largest_share_penalty

  save_model_selection_csv:
    enabled: true

  hdbscan_sweep:
    enabled: false

  llama_resolution_review:
    enabled: false
```

## Output

```text
best_resolution
best_communities
model_selection_rows
```

Saved:

```text
*_cluster_model_selection.csv
```


# Step 20: Assign Final Cluster Numbers Back To Every Review

## What We Do

Map:

```text
microcluster_id -> final_cluster_number
unique review -> final_cluster_number
original row -> final_cluster_number
```

## Technique Used

Python dictionary mapping and `pandas.merge()`.

```python
micro_to_final = {}

for final_cluster_id, community in enumerate(best_communities):
    for microcluster_id in community:
        micro_to_final[microcluster_id] = final_cluster_id

unique_reviews["cluster_number"] = (
    unique_reviews["microcluster_id"]
    .map(micro_to_final)
    .astype(int)
)

clustered_reviews = non_hr_reviews.merge(
    unique_reviews[
        [
            "review_text_normalized",
            "cluster_number",
        ]
    ],
    on="review_text_normalized",
    how="left",
)
```

Optional: sort cluster IDs by size.

```python
cluster_sizes = (
    clustered_reviews["cluster_number"]
    .value_counts()
    .sort_values(ascending=False)
)

cluster_id_remap = {
    old_id: new_id
    for new_id, old_id in enumerate(cluster_sizes.index)
}
```

## Toggle

```yaml
cluster_assignment:
  map_back_to_original_rows: true
  sort_cluster_ids_by_size: true
  preserve_microcluster_id: true
  preserve_unique_review_id: true
```

## Output

```text
clustered_reviews
```

Important columns:

```text
review_text_original
review_text_normalized
platform
source_field
sentiment_side
Classification
microcluster_id
cluster_number
```


# Step 21: Create Cluster Summaries And Human-Readable Labels

## What We Do

Create readable summaries for each cluster:

```text
cluster size
dominant seed theme
top terms
top noun phrases
representative reviews
optional LLM label
```

## Techniques Used

| Technique | Library / Method | Why |
|---|---|---|
| Cluster size | `pandas.groupby()` | Prioritize large clusters |
| Dominant seed theme | weighted average seed similarity | Interpretable theme |
| Top TF-IDF terms | weighted TF-IDF centroid | Keyword evidence |
| Top noun phrases | spaCy noun chunks + `Counter` | Readable phrases |
| NLTK collocations | bigram/trigram extraction | Optional phrase evidence |
| Representative reviews | centroid similarity + heuristics | Real examples |
| Llama labels | prompt top terms and examples | Optional business labels |

## Default Summary Fields

```text
cluster_number
review_count
unique_review_count
dominant_seed_cluster
top_terms
top_noun_phrases
representative_reviews
```

## Representative Review Scoring

```text
representative_score =
centroid_similarity
+ theme_overlap
+ text_richness_score
+ duplicate_count_bonus
- low_information_penalty
```

## Toggle

```yaml
cluster_summary:
  size_summary:
    enabled: true

  dominant_seed_theme:
    enabled: true

  top_terms:
    enabled: true
    top_n: 12

  spacy_top_phrases:
    enabled: true
    top_n: 10

  nltk_collocations:
    enabled: false
    top_n: 10

  representative_reviews:
    enabled: true
    top_n: 5
    avoid_low_information: true

  llama_cluster_labeling:
    enabled: false
    model_id: meta-llama/Llama-3.1-8B-Instruct
    run_on: final_clusters
```

## Output

```text
cluster_summary
```


# Final Outputs

The pipeline should write a run-specific output folder:

```text
results/pipeline_runs/<run_id>/
  config_used.yaml
  run_manifest.json
  normalized_reviews.csv
  clustering_inputs/
    pros_gd.csv
    cons_gd.csv
    likes_am.csv
    dislikes_am.csv
  clusters/
    pros_gd_non_hr_clustered.csv
    cons_gd_non_hr_clustered.csv
    likes_am_non_hr_clustered.csv
    dislikes_am_non_hr_clustered.csv
  summaries/
    *_cluster_summary.csv
    *_cluster_model_selection.csv
  reports/
    clustering_report.md
    clustering_report.docx
  ui_data/
    review_trace.csv
    cluster_cards.json
```


# UI Plan

## Page 1: Pipeline Flow

Technique:

```text
Streamlit + run_manifest.json
```

Shows:

```text
which steps ran
which toggles were enabled
which output folder was created
```

## Page 2: Review Trace

Technique:

```text
Streamlit filters + pandas lookup by review_id
```

Shows:

```text
raw text
cleaned text
source field
HR / Non-HR classification
advocacy factor scores
top seed matches
duplicate count
microcluster id
final cluster number
representative reviews from same cluster
optional Llama explanation
```

## Page 3: Cluster Explorer

Technique:

```text
Streamlit tables and charts
```

Shows:

```text
cluster sizes
dominant themes
top terms
top noun phrases
representative reviews
download links
```

## Page 4: Approach Comparison

Technique:

```text
sklearn.metrics:
Adjusted Rand Index
Normalized Mutual Information
homogeneity
completeness
V-measure
```

Shows:

```text
baseline vs LLM embeddings
baseline vs seed-free clustering
baseline vs HDBSCAN
```

