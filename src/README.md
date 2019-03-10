# Read Me
- What is this?
    - It currently reads through the review.json file in Yelp-Dataset, only 100,000 reviews.
    - Makes a csv file of just reviews and its rating
    - Processes the reviews to get just significant words
        - Removes Punctuation, Digits, Stop Words and makes all words lower case
    - Reads the csv file and vectorizes it using bag of words techniques
    - Runs Perceptron Algorithm on the vectorized form
    - Prints Perceptron Accuracy and Classification Report

- How to RUN?
    - You might need to tweak some file paths according to however you have it set up in you machine
    - Read the header of each file to know, if you need to uncomment some imports
    - Just call the run_classifier.py file

WARNING
    - Don't push the dataset, or any csv or json files
    - Push only the code to the src folder
