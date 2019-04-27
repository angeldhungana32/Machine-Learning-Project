# Read Me
- About?
    - It currently reads through the review.json file in Yelp-Dataset, only 100,000 reviews.
    - Makes a csv file of just reviews and its rating
    - Processes the reviews to get just significant words
        - Removes Punctuation, Digits, Stop Words and makes all words lower case
    - Reads the csv file and vectorizes it using bag of words techniques
    
    - Runs Algorithms on the vectorized form
    - Prints Accuracy and Classification Report

- How to RUN?
    - First run the make file, it will make necessary folders, in your terminal, call 'make setup'
    - If you already have review.csv file, move it to subset_reviews folder
    - Next, depending on what data you want:
        - For All reviews, go to  run_all_review_classifier.py and input the number of subsets you want and run it.
        - For Top Users, go to getrun_top_N_reviewers.py and get N number of top reviewers, and then got to run_top_reviewers and run it
        - For Business, do the same thing as above in run_business.py file
    - Read the header of each file for more information

WARNING
    - Don't push the dataset, or any csv or json files
    - Push only the code to the src folder