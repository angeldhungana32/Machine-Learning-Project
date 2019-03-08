'''
    @author - Angel Dhungana
    This is where we will run our classifiers
    Main File
'''
import process_json
import vectorization
from sklearn.model_selection import train_test_split
import classifier_perceptron
import read_csv
import plot_basic_info_of_data
import scipy
import pickle


def main():
    process_json_review_csv()
    process_vectorization()
    X = open_pickle("x_data.pickle")
    Y = open_pickle("y_data.pickle")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)
    classifier_perceptron.Perceptron_Classifier1(X_train, y_train, X_test,
                                                 y_test)
    classifier_perceptron.Perceptron_Classifier2(X_train, y_train, X_test,
                                                 y_test)


def process_json_review_csv():
    '''
            Make CSV file from JSON File for the review using fields, Text and Stars
    '''
    json_file = 'yelp_dataset/review.json'
    make_this_csv = "review.csv"
    fields = ["text", "stars"]
    max_subset = 100000  # For testing, for actual classifier run we want this the num of reviews
    process_text_what = True  # Do we want to process text?
    prc_jsn = process_json.process_json()
    prc_jsn.make_csv_from_Json(json_file, make_this_csv, fields, max_subset,
                               process_text_what)


def process_vectorization():
    '''
        Vectorize words, and split them into train and test data
    '''
    csv_file_name = "review.csv"
    X_y_data = read_csv.read_csv_file(csv_file_name)
    plot_basic_info_of_data.plt_info(X_y_data[1])
    vec = vectorization.Vectorization()
    vec.vectorize(X_y_data[0])
    pickle.dump(X_y_data[1], open("y_data.pickle", "wb"))


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