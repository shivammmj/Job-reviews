# Review Pipeline Report

## 1. What We Are Building

We are building a reusable Python pipeline that takes employee review data and prepares it for analysis.

The original data comes from two platforms:

- Glassdoor
- AmbitionBox

Each platform stores reviews slightly differently. Glassdoor has `Pros` and `Cons`. AmbitionBox has `Likes` and `Dislikes`.

The goal of this pipeline is to convert those raw review files into clean, structured output files where every review-side text is classified as either:

- `HR`
- `Non HR`

This is the first major stage before final clustering. Once we have clean `Non HR` reviews, those can be passed into the clustering stage separately.

Simple analogy:

Think of the raw dataset like a large mixed box of papers. Some papers are positive comments, some are negative comments, some are about salary and managers, and some are about products or projects. This pipeline first separates the papers into smaller piles, then reads the words, then decides which papers belong to HR-related topics.

## 2. Why We Needed A Pipeline

Earlier, the process was spread across different files, notebooks, and manual steps.

The older flow looked like this:

- Run one script to split reviews.
- Run notebook code to get frequent keywords.
- Run another script to create seed categories.
- Run another script to classify reviews.
- Manually track input and output files.

This is risky because:

- It is easy to run the files in the wrong order.
- It is hard to repeat the same result.
- It is difficult to run on Kaggle or another machine.
- It is difficult to debug which step created which output.

The new flow uses one main command:

```bash
python pipeline/main.py
```

That one command calls the smaller modules in the correct order.

## 3. Iterations Till Now

### Iteration 1: Understand The Existing Work

We first looked at the existing project idea and the reference GitHub repository.

The older repository had scripts like:

- `split_pros_cons copy.py`
- `category_keyword_extension_final.py`
- `classification final.py`
- notebook-style keyword extraction logic

Those files gave the basic direction:

- Split reviews into positive and negative sides.
- Extract keywords from the corpus.
- Expand seed categories using similarity.
- Classify review text as HR or Non HR.

Problem:

The scripts were not modular. They had hard-coded paths and were not easy to run as one pipeline.

### Iteration 2: Decide The Final Input And Output Shape

We decided that the pipeline should start from two raw files:

```text
pipeline/data/raw/all_glassdoor_reviews.csv
pipeline/data/raw/all_ambitionbox_reviews.csv
```

Then it should produce four intermediate files:

```text
pipeline/data/intermediate/pros_gd.csv
pipeline/data/intermediate/cons_gd.csv
pipeline/data/intermediate/likes_am.csv
pipeline/data/intermediate/dislikes_am.csv
```

These four files are important because they preserve the platform meaning:

- Glassdoor `Pros` are positive-side reviews.
- Glassdoor `Cons` are negative-side reviews.
- AmbitionBox `Likes` are positive-side reviews.
- AmbitionBox `Dislikes` are negative-side reviews.

This does not mean every positive comment is good for HR or every negative comment is bad for HR. It only tells us which side of the review the user wrote the comment in.

### Iteration 3: Clarify Positive And Negative Meaning

We clarified that positive and negative are coming directly from the dataset structure.

For example:

- If a user writes inside `Pros`, the platform treats it as a positive-side comment.
- If a user writes inside `Cons`, the platform treats it as a negative-side comment.

This helps analysis because we can later compare:

- What do employees praise?
- What do employees complain about?
- Are HR-related topics more common in complaints or praise?

Example:

`Good work culture and supportive managers` from `Pros` is likely positive HR-related feedback.

`Poor work-life balance and no salary growth` from `Cons` is likely negative HR-related feedback.

Both can be HR-related, but the review side gives us direction.

### Iteration 4: Convert Manual Scripts Into Modules

Instead of keeping separate scripts that must be manually run, we created a `pipeline/` folder.

Each module now handles one small job:

- `split_reviews.py` splits raw files into four files.
- `nlp_keyword_extraction.py` extracts frequent keywords.
- `seed_expansion.py` creates `seed.json`.
- `classify_reviews.py` assigns HR or Non HR labels.
- `main.py` runs all steps in order.

This makes the project easier to understand and easier to extend.

### Iteration 5: Add Toggle-Based Classification Approaches

We added toggles so the same pipeline can classify reviews in different modes.

Important update:

The seed vocabulary is now fixed to the exact paper-derived words. The pipeline no longer expands seed terms from corpus keywords.

Fast mode:

```bash
python pipeline/main.py --seed-backend paper_exact --classification-backend tfidf
```

This uses exact paper seeds and TF-IDF classification. It is good for testing.

Embedding mode:

```bash
python pipeline/main.py --seed-backend paper_exact --classification-backend sentence_transformer
```

This uses exact paper seeds and SentenceTransformer embeddings for classification.

Auto mode:

```bash
python pipeline/main.py
```

This tries embedding-based classification first and falls back to TF-IDF if embeddings are unavailable.

### Iteration 6: Add Dependency And Kaggle Support

We created dependency files:

```text
pipeline/requirements.txt
pipeline/requirements-optional.txt
```

The main requirements file contains the core dependencies.

The optional requirements file contains UMAP and HDBSCAN for later clustering/visualization stages.

This split is useful because `hdbscan` can sometimes be harder to install on Kaggle. The first HR/Non-HR classification pipeline does not need it.

### Iteration 7: Run A Smoke Test

We tested the pipeline using 100 rows.

Command used:

```bash
python3 pipeline/main.py --max-rows 100 --seed-backend paper_exact --classification-backend tfidf
```

The test completed successfully.

Smoke test output summary:

| File | Rows | Non HR Rows |
|---|---:|---:|
| `pros_gd.csv` | 100 | 68 |
| `cons_gd.csv` | 100 | 86 |
| `likes_am.csv` | 99 | 64 |
| `dislikes_am.csv` | 95 | 85 |

This confirmed that:

- Data splitting works.
- Keyword extraction works.
- Seed creation works.
- Classification works.
- Output files are created correctly.

### Iteration 8: Push Pipeline To GitHub

We pushed only the clean pipeline code to GitHub.

Raw datasets were not pushed.

This is important because:

- Raw data files are large.
- GitHub should mainly contain code, documentation, and configuration templates.
- Each user can place their dataset locally or in Kaggle input storage.

GitHub path:

```text
https://github.com/shivammmj/Job-reviews/tree/main/pipeline
```

Commit pushed:

```text
51e3f18 Add modular review pipeline
```

## 4. Current Folder Structure

The current pipeline folder contains:

```text
pipeline/
  .env.example
  .gitignore
  HOW_TO_USE.md
  report.md
  __init__.py
  main.py
  config.py
  utils.py
  split_reviews.py
  nlp_keyword_extraction.py
  seed_expansion.py
  classify_reviews.py
  requirements.txt
  requirements-optional.txt
```

Runtime files are ignored:

```text
pipeline/data/
pipeline/.env
pipeline/__pycache__/
```

## 5. Main Pipeline Flow

The pipeline flow is:

```text
Raw Glassdoor CSV
Raw AmbitionBox CSV
        |
        v
Split into 4 review-side files
        |
        v
Extract frequent keywords
        |
        v
Create seed.json category dictionary
        |
        v
Classify each review as HR or Non HR
        |
        v
Write 4 classified output CSV files
```

## 6. Step 1: Configuration

File:

```text
pipeline/config.py
```

Purpose:

This file stores all important settings in one place.

It reads settings from:

- command-line arguments
- `.env`
- default values

Examples of settings:

- where raw files are stored
- where output files should be written
- whether to use TF-IDF or SentenceTransformer
- classification threshold
- maximum rows for testing
- model name

Why this is useful:

We do not want paths and model names hard-coded inside every script. If everything is in config, we can change behavior without rewriting code.

Example:

```bash
python pipeline/main.py --max-rows 100
```

This changes only the row count for testing.

## 7. Step 2: Split Raw Reviews

File:

```text
pipeline/split_reviews.py
```

Input:

```text
all_glassdoor_reviews.csv
all_ambitionbox_reviews.csv
```

Output:

```text
pros_gd.csv
cons_gd.csv
likes_am.csv
dislikes_am.csv
```

What happens:

The module reads the two raw CSV files using pandas.

Glassdoor columns:

- `Pros`
- `Cons`

AmbitionBox columns:

- `Likes`
- `Dislikes`

The module creates one row per review-side text and keeps useful metadata like:

- company
- rating
- date
- job title
- review title
- employment type
- platform
- sentiment side

Why this step matters:

The two platforms use different column names. This step makes them consistent.

Example:

Glassdoor row:

| Company | Pros | Cons |
|---|---|---|
| Microsoft | Good work culture | Slow promotion |

Becomes:

| output file | review_text | sentiment_side |
|---|---|---|
| `pros_gd.csv` | Good work culture | positive |
| `cons_gd.csv` | Slow promotion | negative |

## 8. Step 3: Keyword Extraction

File:

```text
pipeline/nlp_keyword_extraction.py
```

Input:

```text
pros_gd.csv
cons_gd.csv
likes_am.csv
dislikes_am.csv
```

Output:

```text
keywords.csv
```

Technique used:

- `CountVectorizer` from scikit-learn
- English stopword removal
- n-grams
- frequency counting

What is an n-gram:

An n-gram is a group of words.

Examples:

- 1-gram: `salary`
- 2-gram: `work culture`
- 3-gram: `work life balance`

Why n-grams matter:

Some review topics cannot be understood from one word alone.

Example:

`life` alone is vague.

`work life balance` is meaningful.

The pipeline currently supports up to 3-word phrases by default.

Optional NLP enrichment:

The pipeline also has a spaCy toggle.

If enabled, spaCy can use:

- lemmatization
- POS tagging
- named entity recognition

This can help reduce noise.

Example:

- `managers`, `manager`, `managed` can be normalized better.
- names of organizations or locations can be detected.

Current default:

```text
USE_SPACY=false
```

Reason:

Basic mode is faster and easier to run on Kaggle. spaCy can be enabled later when the environment is ready.

## 9. Step 4: Seed Category Creation

File:

```text
pipeline/seed_expansion.py
```

Input:

```text
keywords.csv
```

Output:

```text
seed.json
```

What is a seed:

A seed is a starting keyword that represents a topic.

Example:

For salary and benefits:

```text
salary
pay
compensation
benefits
bonus
insurance
```

These seed words help the system understand what each category means.

Current paper-derived categories:

- `economic_psychological_benefits`
- `workplace_environment_culture_relationships`
- `product_belief`
- `job_satisfaction`
- `justice`
- `employee_brand_identification`
- `employee_brand_love`
- `inner_self_expressive_brands`
- `brand_reputation`
- `brand_authenticity`
- `social_self_expressive_brands`

What happens:

The pipeline writes the exact paper vocabulary directly into `seed.json`.

Why this matters:

These words are taken from the paper framework, so the categories stay theoretically grounded and are not changed by the current dataset.

Important distinction:

- `keywords.csv` is still created for corpus understanding.
- `seed.json` is not expanded from `keywords.csv`.
- Every seed term has source `paper_seed`.

TF-IDF mode:

This compares reviews against the exact paper seed categories based on token overlap.

Embedding mode:

This compares reviews against the exact paper seed categories using vector embeddings.

Example:

TF-IDF may understand `salary increment` because it shares `salary`.

Embeddings may also understand that `pay raise` is related even if it does not use the exact same words.

## 10. Step 5: Review Classification

File:

```text
pipeline/classify_reviews.py
```

Input:

```text
pros_gd.csv
cons_gd.csv
likes_am.csv
dislikes_am.csv
seed.json
```

Output:

```text
pros_gd_classified.csv
cons_gd_classified.csv
likes_am_classified.csv
dislikes_am_classified.csv
```

What happens:

For each review, the pipeline checks how close the review text is to each category.

It creates score columns like:

```text
score_workplace_environment_culture_relationships
score_economic_psychological_benefits
score_job_satisfaction
```

Then it calculates an `hr_score`.

Current default HR categories:

```text
workplace_environment_culture_relationships
economic_psychological_benefits
```

This means a review is considered HR-related if it strongly talks about:

- culture
- managers
- people
- salary
- benefits
- stress
- work environment

The final label is stored in:

```text
classification
```

Possible values:

- `HR`
- `Non HR`

Example:

Review:

```text
Good salary and supportive manager
```

Likely result:

```text
classification = HR
```

Review:

```text
Product has good architecture and modern tools
```

Likely result:

```text
classification = Non HR
```

Important:

This classification is a rule-guided NLP classification. It is not yet a fully supervised machine learning model trained on manually labeled examples.

## 11. TF-IDF Vs Embedding Mode

The pipeline supports two main styles of text matching.

### TF-IDF

TF-IDF checks important words and phrases.

Good for:

- fast testing
- Kaggle smoke runs
- no model downloads
- explainable output

Limitation:

It depends more on exact words.

Example:

It can easily connect `salary` with `salary increment`.

It may not always connect `pay raise` with `compensation` unless similar words appear.

### SentenceTransformer Embeddings

Embeddings convert text into numerical meaning vectors.

Good for:

- semantic matching
- understanding similar meaning
- better handling of different wording

Limitation:

- needs model download
- slower than TF-IDF
- needs more memory

Example:

Embeddings can understand that these are similar:

```text
good salary
better compensation
pay is strong
```

## 12. Thresholds

The pipeline uses thresholds to decide whether a review is HR or Non HR.

Embedding threshold:

```text
CLASSIFICATION_THRESHOLD=0.50
```

TF-IDF threshold:

```text
TFIDF_CLASSIFICATION_THRESHOLD=0.10
```

Why two thresholds:

TF-IDF scores are usually smaller than embedding scores.

If we use the same threshold for both, TF-IDF may mark almost everything as Non HR.

That happened during the smoke test. So we separated the thresholds.

## 13. Command Line Toggles

Run full default pipeline:

```bash
python pipeline/main.py
```

Run only 100 rows:

```bash
python pipeline/main.py --max-rows 100
```

Run fast TF-IDF mode:

```bash
python pipeline/main.py --seed-backend paper_exact --classification-backend tfidf
```

Run embedding mode:

```bash
python pipeline/main.py --seed-backend paper_exact --classification-backend sentence_transformer
```

Skip splitting if split files already exist:

```bash
python pipeline/main.py --skip-split
```

Skip keyword extraction if `keywords.csv` already exists:

```bash
python pipeline/main.py --skip-keywords
```

Skip seed creation if `seed.json` already exists:

```bash
python pipeline/main.py --skip-seed
```

Skip classification:

```bash
python pipeline/main.py --skip-classification
```

## 14. Dependency Files

Core dependencies:

```text
pipeline/requirements.txt
```

Used for:

- pandas
- numpy
- scikit-learn
- sentence-transformers
- transformers
- torch
- python-dotenv
- tqdm
- spaCy
- NLTK

Optional dependencies:

```text
pipeline/requirements-optional.txt
```

Used for later:

- UMAP visualization
- HDBSCAN density clustering

Why optional:

UMAP and HDBSCAN are useful for clustering and visualization, but they are not required for the first HR/Non-HR classification stage.

## 15. GitHub Push Summary

We pushed the pipeline to:

```text
https://github.com/shivammmj/Job-reviews/tree/main/pipeline
```

We included:

- Python modules
- usage guide
- dependency files
- environment template
- pipeline `.gitignore`

We excluded:

- raw CSV datasets
- `.env`
- generated outputs
- cache files
- Python compiled files

Reason:

GitHub should store reusable code and documentation, not local runtime data.

## 16. What The Pipeline Produces

Intermediate files:

```text
pipeline/data/intermediate/pros_gd.csv
pipeline/data/intermediate/cons_gd.csv
pipeline/data/intermediate/likes_am.csv
pipeline/data/intermediate/dislikes_am.csv
pipeline/data/intermediate/keywords.csv
pipeline/data/intermediate/seed.json
```

Final classified files:

```text
pipeline/data/outputs/pros_gd_classified.csv
pipeline/data/outputs/cons_gd_classified.csv
pipeline/data/outputs/likes_am_classified.csv
pipeline/data/outputs/dislikes_am_classified.csv
```

Run report:

```text
pipeline/data/reports/pipeline_run_manifest.json
```

The manifest stores:

- run time
- config used
- input/output paths
- generated file locations

## 17. What Is Not Done Yet

The current pipeline stops after HR/Non-HR classification.

The next major stage is clustering.

That means:

- Take only `Non HR` rows.
- Generate embeddings.
- Reduce dimensionality with UMAP.
- Cluster using HDBSCAN or another method.
- Compare clustering results with earlier `cluster_review_files.py` outputs.
- Optionally use Llama models to label or validate clusters.

## 18. Suggested Next Steps

Recommended next work:

1. Add a clustering module.
2. Use only `Non HR` rows from the four classified files.
3. Generate embeddings using SentenceTransformer.
4. Use UMAP for 2D visualization.
5. Use HDBSCAN for organic clustering.
6. Save four clustered CSV files.
7. Compare new pipeline clustering with previous clustering outputs.
8. Add charts and cluster summary reports.

## 19. Simple End-To-End Example

Input review:

```text
The salary is good and managers are supportive.
```

Step-by-step:

1. The review comes from either Pros, Cons, Likes, or Dislikes.
2. It is cleaned.
3. Keywords like `salary`, `managers`, and `supportive` are detected.
4. These words match HR seed categories.
5. The review gets high HR-related score.
6. Final output becomes `HR`.

Another input review:

```text
The product uses modern tools and the architecture is scalable.
```

Step-by-step:

1. The review is cleaned.
2. Keywords like `product`, `tools`, and `architecture` are detected.
3. These words match product/work delivery more than HR.
4. HR score stays low.
5. Final output becomes `Non HR`.

## 20. Final Summary

Till now, we converted a manual, multi-file process into a reusable pipeline.

The pipeline can:

- read two raw review datasets
- split them into four review-side files
- extract frequent keywords
- build seed categories
- classify each review as HR or Non HR
- write clean output CSV files
- run locally or on Kaggle
- switch between TF-IDF and embedding-based methods

This gives a stable base for the next stage: clustering only the `Non HR` review rows.
