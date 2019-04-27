import json
import csv
from tqdm import tqdm
import sys
import scipy
from sklearn.model_selection import train_test_split
import classifiers.classifier_perceptron as Perceptron
import numpy as np
'''
    Get the parking data and make a matrix of it.
'''


def main():
    parse_business_parking_data()
    file_name = "business_parking/all_data.csv"
    X, Y = [], []
    with open(file_name) as f:
        readCSV = csv.reader(f)
        for each_row in readCSV:
            Y.append(each_row[0])
            X.append(each_row[2:])
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype(np.float)
    Y = Y.astype(np.float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=101)
    Perceptron.Perceptron_Classifier_3_positive(X_train, y_train, X_test,
                                                y_test)
    Perceptron.Perceptron_Classifier_3_negative(X_train, y_train, X_test,
                                                y_test)


def parse_business_parking_data():
    file_name = "yelp_dataset/business.json"
    output_path = "business_parking/all_data.csv"
    rows = []
    business_data = []
    with open(file_name) as f:
        for line in f:
            business_data.append(json.loads(line))
    f.close()

    for entry in tqdm(range(0, len(business_data))):
        row = []
        row.append(float(business_data[entry]['stars']))

        # extract the parking attributes
        parking_attributes = ['street', 'valet', 'lot', 'garage', 'validated']
        # for each parking attribute
        for attribute in parking_attributes:
            # if there are parking attributes
            if business_data[entry][
                    'attributes'] != None and 'BusinessParking' in business_data[
                        entry]['attributes']:
                if attribute in business_data[entry]['attributes'][
                        'BusinessParking']:
                    # if the parking attribute is true, 1 else -1
                    information = business_data[entry]['attributes'][
                        'BusinessParking']
                    information = information[1:-1]
                    load_attribute = get_attr_res(information)
                    if load_attribute[attribute] is True:
                        row.append(1)
                    elif load_attribute[attribute] is False:
                        row.append(0)
                else:
                    row.append(-1)
            # else if the parking attribute is not available
            else:
                row.append(-1)
        rows.append(row)

    with open(output_path, 'w') as out:
        writer = csv.writer(out)
        for entry in tqdm(range(0, len(rows))):
            try:
                writer.writerow(rows[entry])
            except UnicodeEncodeError:
                continue
    out.close()


def get_attr_res(information):
    '''
        Helper Function that gets the parking information
    '''
    split_inf = information.split(',')
    dictionary_attr = {}
    for each_x in split_inf:
        next_split = each_x.split(':')
        key = next_split[0].strip()
        if next_split[1].strip() == 'True':
            value = True
        else:
            value = False
        dictionary_attr[key.strip('\'')] = value
    return dictionary_attr


if __name__ == "__main__":
    main()