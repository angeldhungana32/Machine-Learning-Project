import json
import csv
from tqdm import tqdm
import sys
import scipy
import plot_basic_info_of_data
from sklearn.model_selection import train_test_split
import classifiers.classifier_neuralnetworks as NN
import classifiers.classifier_logisticregression as LR
import classifiers.classifier_perceptron as P
import classifiers.classifier_randomforest as RF
import classifiers.classifier_stochasticgd as SGD
import classifiers.classifier_bagging as BG
import numpy as np
import copy
import math
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
    #X, Y = remove(X, Y)
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype(np.float)
    Y = Y.astype(np.float)
    Y = Y.astype(np.int)
    get_count_and_plot(Y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=101, shuffle=True)
    print("Running Classifiers")
    LR.Logistic_Regression("results/logistic_regression_business.txt", X_train,
                           y_train, X_test, y_test)
    NN.Neural_Network("results/neural_networks_business.txt", X_train, y_train,
                      X_test, y_test)
    RF.Random_Forest("results/random_forest_business.txt", X_train, y_train,
                     X_test, y_test)
    SGD.Stochastic_Gradient_Descent("results/stochastic_gradient_business.txt",
                                    X_train, y_train, X_test, y_test)
    BG._Bagging("results/bagging_business.txt", X_train, y_train, X_test,
                y_test)
    P._Perceptron("results/perceptron_business.txt", X_train, y_train, X_test,
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


def remove(X, Y):
    '''
        Only includes available datat 
    '''
    XX = []
    YY = []
    for i in range(len(X)):
        if X[i][0] != '-1':
            XX.append(X[i])
            YY.append(Y[i])
    return XX, YY


def get_count_and_plot(y_data):
    '''
        Get the rating based on 
    '''
    ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for y in y_data:
        ratings[int(round(y))] += 1
    stars = [ratings[1], ratings[2], ratings[3], ratings[4], ratings[5]]
    plot_basic_info_of_data.plt_stars(stars, "Ratings Chart Business Parking",
                                      "business_parking/stars.pdf")


if __name__ == "__main__":
    main()
