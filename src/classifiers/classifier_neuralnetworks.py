from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
'''
   Runs the MultiLayer Perceptron (NN) Classifier
'''


def NN_Classifier_3_positive(X_train, y_train, X_test, y_test):
    '''
        Make binary classification of positive or negative
        Considering star rating 3 as positive
        MultiLayer Perceptron NN Classifier
    '''
    nn = MLPClassifier(
        solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # Change the ratings to
    new_y_train1 = process_Y1(y_train)
    new_y_test1 = process_Y1(y_test)
    nn.fit(X_train, new_y_train1)
    y_pred = nn.predict(X_test)
    print()
    print("Star Rating 3 considered as positive")
    print()
    print('Perceptron Accuracy: %.2f' % accuracy_score(new_y_test1, y_pred))
    print()
    print('Classification Report:')
    print(classification_report(new_y_test1, y_pred))


def process_Y1(y_):
    '''
        Considering 3 as positive
    '''
    newY = []
    for i in range(len(y_)):
        if float(y_[i]) < 3.0: newY.append(-1)
        else: newY.append(1)
    return newY


def NN_Classifier_3_negative(X_train, y_train, X_test, y_test):
    '''
        Make binary classification of positive or negative
        Considering star rating 3 as positive
        Run the MultiLayer Perceptron Classifier and Print Accuracy
    '''
    nn = MLPClassifier(
        solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    new_y_train2 = process_Y2(y_train)
    new_y_test2 = process_Y2(y_test)
    nn.fit(X_train, new_y_train2)
    y_pred = nn.predict(X_test)
    print()
    print("Star Rating 3 considered as negative")
    print()
    print('Perceptron Accuracy: %.2f' % accuracy_score(new_y_test2, y_pred))
    print()
    print('Classification Report:')
    print(classification_report(new_y_test2, y_pred))


def process_Y2(y_):
    '''
       Considering 3 as negative
    '''
    newY = []
    for i in range(len(y_)):
        if float(y_[i]) <= 3.0: newY.append(-1)
        else: newY.append(1)
    return newY
