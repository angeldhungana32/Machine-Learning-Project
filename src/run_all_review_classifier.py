import process_json
import vectorization
from sklearn.model_selection import train_test_split
import read_csv
import plot_basic_info_of_data
import scipy
import pickle
# Import classifiers like this
import classifiers.classifier_perceptron as Perceptron
import classifiers.classifier_adaboost as Aboost
'''
    Provide the Number of reviews you want, and then run the classifier you want
    Max Number of Reviews = 6,685,900
    Default Number of Review Subset = 1,000,000
'''


def main():
    max_subset = 10000
    #process_json_review_csv(max_subset)
    print('vectorizing')
    process_vectorization()
    X = open_pickle("subset_reviews/x_data.pickle")
    Y = open_pickle("subset_reviews/y_data.pickle")
    print('split')
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)
    # Run the Classifier you want here
    #Perceptron.Perceptron_Classifier_3_positive(X_train, y_train, X_test, y_test)
    #Perceptron.Perceptron_Classifier_3_negative(X_train, y_train, X_test, y_test)
    Aboost.boost(310, 0.7, X_train, y_train, X_test, y_test)
            

        

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
    plot_basic_info_of_data.plt_info(X_y_data[1], "subset_reviews/stars.pdf")
    vec = vectorization.Vectorization()
    vec.vectorize(X_y_data[0], "subset_reviews/x_data.pickle")
    pickle.dump(X_y_data[1], open("subset_reviews/y_data.pickle", "wb"))


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