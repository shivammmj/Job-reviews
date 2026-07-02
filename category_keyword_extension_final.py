import pandas as pd
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import defaultdict, Counter
import spacy

# -----------------------------
# THIS CAN RUN IN BATCHES AND WILL UPDATE THE JSON FILE IF RAN MORE THAN ONCE
# -----------------------------
BATCH_SIZE = 10000
CHECKPOINT_FILE = "Classification Data/batch_checkpoint.txt"
OUTPUT_FILE = "Classification Data/seed.json"
WINDOW_SIZE = 2
MIN_COOC_FREQ = 5
threshold = 0.6

# -----------------------------
# LOAD MODELS
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# LOAD CORPUS OF TERMS
# -----------------------------
df = pd.read_csv("Classification Data/keywords.csv")  # change filename
corpus = df["term"].dropna().astype(str).str.lower().unique().tolist()


import re

def clean_text(text):
    text = str(text)

    # Remove newline, tabs
    text = text.replace("\n", " ").replace("\t", " ")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters (keep alphabets + space)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

# -----------------------------
# CATEGORIES
# -----------------------------
categories = {
    "workplace_environment_culture_relationships": [
        "community", "co-worker", "culture", "workplace environment",
        "collaboration", "friendly", "atmosphere"
    ],
    "economic_psychological_benefits": [
        "work-life balance", "wellbeing", "monetary benefits", "goodies"
    ],
    "job_satisfaction": [
        "meaningful work", "job satisfaction", "gratification"
    ],
    "justice": [
        "fair", "justice", "just"
    ],
    "employee_brand_identification": [
        "feel like family", "we", "our", "second home"
    ],
    "brand_reputation": [
        "great brand", "respected brand", "valued brand", "good brand name"
    ],
    "product_belief": [
        "products", "quality"
    ],
    "employee_brand_love": [
        "love", "like", "passionate"
    ],
    "inner_self_expressive_brands": [
        "share my belief", "values align with", "my personality"
    ],
    "brand_authenticity": [
        "genuine", "honest", "transparency", "trust", "accountability", "authentic"
    ],
    "social_self_expressive_brands": [
        "feel important", "people think", "tell people"
    ]
}

# -----------------------------
# CATEGORY EMBEDDINGS (ONCE)
# -----------------------------
category_embeddings = {
    cat: model.encode(words)
    for cat, words in categories.items()
}

category_centroids = {
    cat: np.mean(embs, axis=0)
    for cat, embs in category_embeddings.items()
}

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        last_completed_batch = int(f.read().strip())
else:
    last_completed_batch = -1

# -----------------------------
# LOAD EXISTING JSON (if exists)
# -----------------------------
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        global_expansions = json.load(f)
else:
    global_expansions = {cat: [] for cat in categories}

# Convert to dict for deduplication
global_dict = {
    cat: {item["word"]: item["score"] for item in words}
    for cat, words in global_expansions.items()
}

# -----------------------------
# BATCH PROCESSING
# -----------------------------
# -----------------------------
# LOAD DATA (TERMS DIRECTLY)
# -----------------------------

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        last_completed_batch = int(f.read().strip())
else:
    last_completed_batch = -1

# -----------------------------
# LOAD EXISTING JSON
# -----------------------------
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        global_expansions = json.load(f)
else:
    global_expansions = {cat: [] for cat in categories}

global_dict = {
    cat: {item["word"]: item["score"] for item in words}
    for cat, words in global_expansions.items()
}

# -----------------------------
# BATCH PROCESSING
# -----------------------------
num_batches = (len(corpus) // BATCH_SIZE) + 1

for batch_num in range(num_batches):

    if batch_num <= last_completed_batch:
        print(f"Skipping batch {batch_num + 1}")
        continue

    print(f"\nProcessing batch {batch_num + 1}/{num_batches}...")

    start = batch_num * BATCH_SIZE
    end = start + BATCH_SIZE
    batch_terms = corpus[start:end]

    # EMBEDDINGS
    term_embeddings = model.encode(batch_terms, batch_size=64)

    for i, term in enumerate(batch_terms):
        term_vec = term_embeddings[i].reshape(1, -1)

        best_cat = None
        best_score = -1

        for cat, centroid in category_centroids.items():
            score = cosine_similarity(
                term_vec,
                centroid.reshape(1, -1)
            )[0][0]

            if score > best_score:
                best_score = score
                best_cat = cat

        if best_score >= threshold:
            if term in global_dict[best_cat]:
                global_dict[best_cat][term] = max(
                    global_dict[best_cat][term],
                    best_score
                )
            else:
                global_dict[best_cat][term] = best_score

    # Add seeds (always keep)
    for cat, seed_words in categories.items():
        for word in seed_words:
            global_dict[cat][word] = 1.0

    # Convert back to sorted list
    final_output = {}
    for cat, word_dict in global_dict.items():
        sorted_words = sorted(
            word_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_output[cat] = [
            {"word": w, "score": float(round(s, 3))}
            for w, s in sorted_words[:100]
        ]

    # SAVE
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Updated: {OUTPUT_FILE}")

    # UPDATE CHECKPOINT
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(batch_num))