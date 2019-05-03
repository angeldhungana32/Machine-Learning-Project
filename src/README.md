# Machine Learning Class Project

## Collaborators 

Angel Dhungana, William Gerichs

## Assignment Overview

A collaborative for project for CS 5350 for Spring 2019.
The objective is to use machine learning algorithms on some real data set, with the goal to gain in depth experience with exploring and solving real world practical problems and demonstrate a deep understanding of the aspects of Machine Learning.

## About

* Machine Learning Project on [Yelp Dataset](https://www.yelp.com/dataset)
* We are trying to predict if we can classify if a user review is positive or negative using various classifiers and how accurate they are.
* We are trying to see if we can predict a business rating based on the reviews of only the top users.
* We are trying to see if we can predict a business rating based on of their attributes, meaning we want to see if having or not having a business attribute like Parking affects the user sentiments as well as the business rating.
* Improving the Adaboosting algorithm on predicting whether the given review was good or bad. Tweaking various parameters to find out which worked the best and gave higher accuracy.
    

## Data Processing
**All Reviews**
- Reads through the review.json file from Yelp Data Set, and gets all the Reviews and its Star Count
- Processes the reviews to just get the significant words
    - Removes Punctuation, Digits, Stop Words and makes all words lower case
- Adds it into a CSV File
- Reads the CSV file and vectorizes it using bag of words techniques
- This results a sparse matrix, thus adds it into a pickle file using CSR Matrix, thus saving the memory
    - All this is done in process_json.py
- Now, we open the Pickle file, and split the data using Cross Validation method, into 70% for Training and 30% for Testing with random shuffle
- We choose the reviews to be positive or negative based on their rating,
    - Two Choices: 
       - &gt; 3 positive, else negative 
        - &gt;= 3 positive else negative
- We run various classifiers located in **classifiers folder** and store the Classification Report in **results folder**
    - This is done in run_all_review_classifier.py file.
    - **This also includes testing the adaboost classifer using various parameter**

**Top Users**
- Reads through the users.json file from Yelp Data Set and gets top N users based on their review count.
    - We store metadata of user's name, their user id and their review count
- Based on the metadata, we read the review.json file and only get the reviews of these particular users
- Then, we process these reviews and run various classifiers similarly as above.
    - File for this is run_top_N_reviewer.py

**Business Parking Attribute**
- Reads through business.json file from the dataset and gets the parking information from the Attributes section.
- The Parking itself itself, has five variables like Street, Validated,... etc.
- If a Business has certain parking, it is labeled as 1 otherwise 0.
- Then, we run the classifiers on this dataset.
    - File for this is run_business_parking_attribute.py

## How to RUN?
* First run the make file, it will make necessary folders, in your terminal, call 'make setup'
* Next, depending on what data you want:
    * For All reviews, go to  **run_all_review_classifier.py** and input the number of subsets you want and run it.
    * For Top Users, go to **run_top_N_reviewer.py** and get N number of top reviewers, and then got to run_top_reviewers and run it
    * For Business, do the same thing as above in **run_business_parking_attribute.py** file
* Read the header of each file for more information
* If you want to change classifier information, go to **classifiers folder** and change as you need.

## NOTE
- The dataset is big so only the src files and the classification reports are included in this Github repository.
