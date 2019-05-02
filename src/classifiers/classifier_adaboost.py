from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
import datetime
'''
    Runs the Ada boost classifier with various parameters
'''


def process_Y1(y_):
    '''
        Considering 3 as positive
    '''
    newY = []
    for i in range(len(y_)):
        if float(y_[i]) < 3.0: newY.append(-1)
        else: newY.append(1)
    return newY


def boost(e, l, X_train, y_train, X_test, y_test):
    """ Make the adaboost and run it printing results
    """
    ada = AdaBoostClassifier(n_estimators=e, learning_rate=l)
    print('changing label')
    print(datetime.datetime.now())
    new_y_train = process_Y1(y_train)
    new_y_test = process_Y1(y_test)
    print('training!')
    print(datetime.datetime.now())
    ada.fit(X_train, new_y_train)

    #prediction
    print('testing')
    print(datetime.datetime.now())
    pred = ada.predict(X_test)
    #print('AdaBoost with ' , str(e) , 'classifiers and ' , str(l) 'learning rate')
    print('Testing Accuracy: ', metrics.accuracy_score(new_y_test, pred))
    #print('Training Accuracy: ', metrics.accuracy_score(new_y_train, pred))
