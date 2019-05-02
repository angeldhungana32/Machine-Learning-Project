import process_json
import vectorization
from sklearn.model_selection import train_test_split
import read_csv
import plot_basic_info_of_data
import classifiers.classifier_neuralnetworks as NN
import classifiers.classifier_logisticregression as LR
import classifiers.classifier_perceptron as P
import classifiers.classifier_randomforest as RF
import classifiers.classifier_stochasticgd as SGD
import classifiers.classifier_bagging as BG
import classifiers.classifier_adaboost as Aboost
import scipy
import pickle
'''
    Provide the Number of reviews you want, and then run the classifier you want
    Max Number of Reviews = 6,685,900
    Default Number of Review Subset = 1,000,000
'''


def main():
    max_subset = 1000000
    process_json_review_csv(max_subset)
    print("Processing Vectorization")
    process_vectorization()
    X = open_pickle("subset_reviews/x_data.pickle")
    Y = open_pickle("subset_reviews/y_data.pickle")
    print("Splitting X and Y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)
    # Run the Classifier you want here
    print("Running Classifiers")
    run_all_classifiers(X_train, X_test, y_train, y_test)
    print("Running AdaBoost Tester")
    Aboost.boost(310, 0.7, X_train, y_train, X_test, y_test)


def run_all_classifiers(X_train, X_test, y_train, y_test):
    '''
        Run different classifiers
    '''
    LR.Logistic_Regression("results/logistic_regression_all.txt", X_train,
                           y_train, X_test, y_test)
    P._Perceptron("results/perceptron_all.txt", X_train, y_train, X_test,
                  y_test)
    NN.Neural_Network("results/neural_networks_all.txt", X_train, y_train,
                      X_test, y_test)
    RF.Random_Forest("results/random_forest_all.txt", X_train, y_train, X_test,
                     y_test)

    SGD.Stochastic_Gradient_Descent("results/stochastic_gradient_all.txt",
                                    X_train, y_train, X_test, y_test)

    BG._Bagging("results/bagging_all.txt", X_train, y_train, X_test, y_test)

    LR.run_various_classifiers(X_train, y_train, X_test, y_test)


def process_json_review_csv(max_subset):
    '''
            Make CSV file from JSON File for the review using fields, Text and Stars
    '''
    json_file = 'yelp_dataset/review.json'
    make_this_csv = "subset_reviews/review.csv"
    fields = ["text", "stars"]
    process_text_what = True  # Do we want to process text?
    prc_jsn = process_json.process_json()
    prc_jsn.make_csv_from_Json(json_file, make_this_csv, fields, max_subset,
                               process_text_what)


def process_vectorization():
    '''
        Vectorize words, and split them into train and test data
    '''
    csv_file_name = "subset_reviews/review.csv"
    X_y_data = read_csv.read_csv_file(csv_file_name)
    plot_basic_info_of_data.plt_info(X_y_data[1], "Ratings Chart All Reviews",
                                     "subset_reviews/stars.pdf")
    vec = vectorization.Vectorization()
    vec.vectorize(X_y_data[0], "subset_reviews/x_data.pickle")
    pickle.dump(X_y_data[1], open("subset_reviews/y_data.pickle", "wb"))


def open_pickle(file_name):
    '''
        Open Pickle file and return the matrix
    '''
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