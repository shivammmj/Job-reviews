import csv
import shutil

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "Dataset/all_ambitionbox_reviews.csv"
PROS_OUTPUT = "Classification Data/likes_am.csv"
CONS_OUTPUT = "Classification Data/dislikes_am.csv"

PROS_COLUMN = "Likes"
CONS_COLUMN = "Dislikes"

# -----------------------------
# 1. Duplicate files
# -----------------------------
shutil.copy(INPUT_CSV, PROS_OUTPUT)
shutil.copy(INPUT_CSV, CONS_OUTPUT)

# -----------------------------
# 2. Function to remove column
# -----------------------------
def remove_column(file_path, column_to_remove, filter_column):
    with open(file_path, "r", newline='', encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = [f for f in reader.fieldnames if f != column_to_remove]

        rows = []
        for row in reader:
            # keep only rows where filter_column is non-empty
            value = row.get(filter_column, "")
            if value and value.strip():
                row.pop(column_to_remove, None)
                rows.append(row)

    # overwrite same file
    with open(file_path, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------
# 3. Process files
# -----------------------------

# Pros file → remove Cons column
remove_column(PROS_OUTPUT, CONS_COLUMN, PROS_COLUMN)

# Cons file → remove Pros column
remove_column(CONS_OUTPUT, PROS_COLUMN, CONS_COLUMN)

print("Done:")
print(f"- {PROS_OUTPUT}")
print(f"- {CONS_OUTPUT}")