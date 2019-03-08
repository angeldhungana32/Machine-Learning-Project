'''
    This is where we will run our classifiers
    Main File
'''
import process_json
import vectorization
from sklearn.model_selection import train_test_split
import classifier_perceptron


def main():
    process_json_review_csv()
    X_train, X_test, y_train, y_test = process_vectorization()
    classifier_perceptron.Perceptron_Classifier(X_train, y_train, X_test,
                                                y_test)


def process_json_review_csv():
    '''
            Make CSV file from JSON File for the review using fields, Text and Stars
    '''
    json_file = 'yelp_dataset/review.json'
    make_this_csv = "review.csv"
    fields = ["text", "stars"]
    max_subset = 10000  # For testing, for actual classifier run we want this the num of reviews
    process_text_what = True  # Do we want to process text?
    prc_jsn = process_json.process_json()
    prc_jsn.make_csv_from_Json(json_file, make_this_csv, fields, max_subset,
                               process_text_what)


def process_vectorization():
    '''
        Vectorize words, and split them into train and test data
    '''
    csv_file_name = "review.csv"
    vec = vectorization.Vectorization()
    X, Y = vec.vectorize(csv_file_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()