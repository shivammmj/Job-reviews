# Job-reviews

Clean the data, and get 4 files pros_gd, cons_gd, likes_am, dislikes_am

Create a single main file which calls the other modules
1. Convert all modules into callable modules (split_pros_cons copy.py,category_keyword_extension_final.py, classification final.py and google collab code )
2. Run the google collab code with 4 files as input to get the frequent keywords from the corpus
3. Run the category_keyword_extension_final.py file with the corpus pointing to keywords.csv to create a seed.json
4. Run the classification final.py code on the 4 files with seed as seed.json created in 3rd step



  The flow will be:

  classified CSV
  -> keep only Non HR rows
  -> convert each review into embedding vector
  -> reduce vectors using UMAP
  -> cluster using HDBSCAN
  -> save clustered CSV

  Low-level explanation:

  1. Read the classified files

  We already have:

  pros_gd_classified.csv
  cons_gd_classified.csv
  likes_am_classified.csv
  dislikes_am_classified.csv

  Each file has a column:

  classification

  We will keep only:

  classification == "Non HR"

  2. Convert reviews into embeddings

  Each review is text. Algorithms cannot directly understand text.

  So we convert each review into numbers using:

  SentenceTransformer

  Example:

  "Good product and interesting technical work"

  becomes something like:

  [0.12, -0.08, 0.44, ...]

  This number vector captures meaning.

  3. Reduce dimensions using UMAP

  Embeddings have many dimensions, usually hundreds.

  UMAP compresses them into fewer dimensions while trying to keep similar reviews close together.

  Why use UMAP:

  high-dimensional embeddings are hard to cluster directly
  UMAP makes the structure easier for clustering

  4. Cluster using HDBSCAN

  HDBSCAN finds natural groups.

  Why HDBSCAN:

  we do not need to manually decide cluster count
  it can detect outliers
  it works well with organic review themes

  Example result:

  cluster 0 = product quality comments
  cluster 1 = technical project comments
  cluster 2 = customer/client work comments
  cluster -1 = outliers/noise

  5. Save 4 clustered output files

  Final output:

  pros_gd_non_hr_clustered.csv
  cons_gd_non_hr_clustered.csv
  likes_am_non_hr_clustered.csv
  dislikes_am_non_hr_clustered.csv

  Each row will contain:

  review_text
  classification
  cluster_id
  umap_x
  umap_y
  primary_category
  hr_score
  metadata columns

  6. Compare with old clustering

  After generating clusters, we compare them with the earlier output from:

  cluster_review_files.py

  Comparison will check:

  number of clusters
  cluster sizes
  outlier count
  top sample reviews per cluster
  common themes

  In short:

  Non HR reviews are grouped by meaning, not by keywords alone.

  We use:

  SentenceTransformer = understand meaning
  UMAP = simplify embedding space
  HDBSCAN = find natural clusters
