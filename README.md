# Job-reviews

Clean the data, and get 4 files pros_gd, cons_gd, likes_am, dislikes_am

Create a single main file which calls the other modules
1. Convert all modules into callable modules (split_pros_cons copy.py,category_keyword_extension_final.py, classification final.py and google collab code )
2. Run the google collab code with 4 files as input to get the frequent keywords from the corpus
3. Run the category_keyword_extension_final.py file with the corpus pointing to keywords.csv to create a seed.json
4. Run the classification final.py code on the 4 files with seed as seed.json created in 3rd step
