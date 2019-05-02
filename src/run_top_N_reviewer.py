import process_json
import vectorization
from sklearn.model_selection import train_test_split
import classifiers.classifier_neuralnetworks as NN
import classifiers.classifier_logisticregression as LR
import classifiers.classifier_perceptron as P
import classifiers.classifier_randomforest as RF
import classifiers.classifier_stochasticgd as SGD
import classifiers.classifier_bagging as BG
import read_csv
import plot_basic_info_of_data
import scipy
import pickle
import csv
import json
'''
    This is where we will run our classifiers for Top Reviewers

    Input the number of top users review you want
        num_of_users = ..., default is 100

    Give the fields you want to extract from JSON to CSV
        fields = [], defaults are User ID, Name and Review Count
        ** Make sure the first and second in the list are always "user_id" and "name"**
    
'''


def main():
    num_of_users = 100
    fields = ["user_id", "name", "review_count"]
    print("Getting top Users Now...")
    process_json_user_review(num_of_users, fields)
    print("Getting user reviews from top Users")
    make_user_csv_from_Json()
    print("Transform bag of words to Vectorized Matrix")
    process_vectorization()

    X = open_pickle("top_reviewers/topreviewers_x.pickle")
    Y = open_pickle("top_reviewers/topreviewers_y.pickle")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)

    # Run any classifier you want
    print("Running Classifiers")
    run_classifiers(X_train, X_test, y_train, y_test)


def run_classifiers(X_train, X_test, y_train, y_test):
    LR.Logistic_Regression("results/logistic_regression_user.txt", X_train,
                           y_train, X_test, y_test)
    P._Perceptron("results/perceptron_user.txt", X_train, y_train, X_test,
                  y_test)
    NN.Neural_Network("results/neural_networks_user.txt", X_train, y_train,
                      X_test, y_test)
    RF.Random_Forest("results/random_forest_user.txt", X_train, y_train,
                     X_test, y_test)
    SGD.Stochastic_Gradient_Descent("results/stochastic_gradient_user.txt",
                                    X_train, y_train, X_test, y_test)
    BG._Bagging("results/bagging_user.txt", X_train, y_train, X_test, y_test)


def process_json_user_review(num_of_users, fields):
    '''
            Make CSV file from JSON File for the review using fields, Text and Stars
    '''
    json_file = 'yelp_dataset/user.json'
    make_this_csv = "top_reviewers/metadata.csv"
    prc_jsn = process_json.process_json()
    prc_jsn.make_metadata_from_user_reviews(json_file, make_this_csv, fields,
                                            num_of_users)


def make_user_csv_from_Json():
    '''
        Make CSV file only for user data
    '''
    json_file = 'yelp_dataset/review.json'
    metadata_file = "top_reviewers/metadata.csv"
    fields = ["text", "stars", "date"]
    prc_jsn = process_json.process_json()
    prc_jsn.make_csv_for_top_users_from_Json(json_file, metadata_file, fields,
                                             True)


def process_vectorization():
    '''
        Vectorize words, and split them into X and Y
    '''
    file_name = "top_reviewers/topreviewers.csv"
    X_y_data = read_csv.read_csv_file(file_name)
    plot_basic_info_of_data.plt_info(X_y_data[1], "Ratings Chart Top Users",
                                     "top_reviewers/stars_users.pdf")
    vec = vectorization.Vectorization()
    x_name = "top_reviewers/topreviewers_x.pickle"
    y_name = "top_reviewers/topreviewers_y.pickle"
    vec.vectorize(X_y_data[0], x_name)
    pickle.dump(X_y_data[1], open(y_name, "wb"))


def open_pickle(file_name):
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects


if __name__ == "__main__":
    main()