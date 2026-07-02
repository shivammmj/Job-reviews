import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

#CSVS which need to be classified
INPUT_CSVS = [
    "Classification Data/dislikes_am.csv",
    "Classification Data/likes_am.csv",
    "Classification Data/pros_gd.csv",
    "Classification Data/cons_gd.csv"
]

#KEYWORD SEED BASED ON WHICH WE CLASSIFY
KEYWORD_JSON = "Classification Data/seed.json"
#NUMBER OF ROWS CLASSIFIED AT ONCE
BATCH_SIZE = 10000
#THRESHOLD = 0.45

# -----------------------------
# 1. Load keyword JSON
# -----------------------------
with open(KEYWORD_JSON, "r") as f:
    keyword_dict = json.load(f)

categories = list(keyword_dict.keys())

# -----------------------------
# 2. Load model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 3. Prepare embeddings
# -----------------------------
category_embeddings = {}
category_weights = {}

for category, items in keyword_dict.items():
    words = [item["word"] for item in items]
    scores = [item["score"] for item in items]

    emb = model.encode(words, normalize_embeddings=True)

    category_embeddings[category] = emb
    category_weights[category] = np.array(scores)

# -----------------------------
# 4. Load CSV
# -----------------------------
for INPUT_CSV in INPUT_CSVS:
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

    # -----------------------------
    # Detect TEXT COLUMN
    # -----------------------------
    TEXT_COLUMN = None
    possible_column_names = ["Cons", "Pros", "Likes", "Dislikes", "review", "phrase"]

    for col in possible_column_names:
        if col in df.columns:
            TEXT_COLUMN = col
            print(f"Using column: {TEXT_COLUMN}")
            break

    if TEXT_COLUMN is None:
        raise ValueError("No valid review column found")

    # -----------------------------
    # Clean TEXT COLUMN
    # -----------------------------
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()

    # -----------------------------
    # 5. Initialize columns
    # -----------------------------
    if "Classification" not in df.columns:
        df["Classification"] = np.nan

    # Fix old CSVs (important)
    df["Classification"] = df["Classification"].replace("", np.nan)

    for cat in categories:
        if cat not in df.columns:
            df[cat] = np.nan

    # -----------------------------
    # 6. Resume logic
    # -----------------------------
    processed_count = df["Classification"].notna().sum()
    print(f"Resuming from row {processed_count}")

    # -----------------------------
    # 7. BATCH LOOP (continuous)
    # -----------------------------
    while processed_count < len(df):

        end_index = min(processed_count + BATCH_SIZE, len(df))

        batch_df = df.iloc[processed_count:end_index]

        batch_sentences = batch_df[TEXT_COLUMN].tolist()

        # Encode batch
        batch_embeddings = model.encode(batch_sentences, normalize_embeddings=True)

        # -----------------------------
        # Compute scores
        # -----------------------------
        all_scores = {cat: [] for cat in categories}

        for category in categories:
            sims = np.dot(batch_embeddings, category_embeddings[category].T)
            weighted = sims * category_weights[category]
            max_scores = np.max(weighted, axis=1)
            all_scores[category] = max_scores

        # -----------------------------
        # Assign results
        # -----------------------------
        for i, idx in enumerate(range(processed_count, end_index)):

            scores = {cat: float(all_scores[cat][i]) for cat in categories}

            # IF SCORES OF THE TWO HR RELATED CATEGORIES IS GREATER THAN 0.5 THEN HR ELSE NON-HR
            score_sum =scores.get("workplace_environment_culture_relationships", 0) + scores.get("economic_psychological_benefits", 0)

            if score_sum > 0.5:
                df.at[idx, "Classification"] = "HR"
            else:
                df.at[idx, "Classification"] = "Non HR"

            # Write category scores
            for cat in categories:
                df.at[idx, cat] = scores[cat]

        # -----------------------------
        # Update progress
        # -----------------------------
        processed_count = end_index

        # Save after each batch (safe)
        df.to_csv(INPUT_CSV, index=False)

        print(f"Processed up to row {processed_count} for {INPUT_CSV}")

    print(f"All rows processed for {INPUT_CSV}")