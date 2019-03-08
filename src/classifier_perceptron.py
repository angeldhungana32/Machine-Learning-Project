'''
    This runs the Perceptron Classifier
'''
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


def Perceptron_Classifier(X_train, y_train, X_test, y_test):
    '''
        Run the Perceptron Classifier and Print Accuracy
    '''
    percep = Perceptron(max_iter=100, eta0=0.1, tol=0.0001, random_state=0)
    percep.fit(X_train, y_train)
    y_pred = percep.predict(X_test)
    print()
    print('Perceptron Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))