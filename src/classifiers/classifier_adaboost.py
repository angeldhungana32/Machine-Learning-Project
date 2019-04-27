from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics

def boost(e, l, X_train, y_train, X_test, y_test):
    """ Make the adaboost and run it printing results
    """
    ada = AdaBoostClassifier(n_estimators=e, learning_rate=l)
    ada.fit(X_train, y_train)

    #prediction
    pred = ada.predict(X_test)
    #print('AdaBoost with ' , str(e) , 'classifiers and ' , str(l) 'learning rate')
    print('Accuracy: ', metrics.accuracy_score(y_test, pred))
