from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
'''
   Runs the Naive Bayes Classifier
'''


def NaiveBayes_Classifier_3_positive(X_train, y_train, X_test, y_test):
    '''
        Make binary classification of positive or negative
        Considering star rating 3 as positive
        Gaussian Naive Bayes Classifier
    '''
    gnb = GaussianNB()
    # Change the ratings to
    new_y_train1 = process_Y1(y_train)
    new_y_test1 = process_Y1(y_test)
    gnb.fit(X_train, new_y_train1)
    y_pred = gnb.predict(X_test)
    print()
    print("Star Rating 3 considered as positive")
    print()
    print('Naive Bayes Accuracy: %.2f' % accuracy_score(new_y_test1, y_pred))
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


def NaiveBayes_Classifier_3_negative(X_train, y_train, X_test, y_test):
    '''
        Make binary classification of positive or negative
        Considering star rating 3 as positive
        Run the Gaussian Naive Bayes and Print Accuracy
    '''
    gnb = GaussianNB()
    new_y_train2 = process_Y2(y_train)
    new_y_test2 = process_Y2(y_test)
    gnb.fit(X_train, new_y_train2)
    y_pred = gnb.predict(X_test)
    print()
    print("Star Rating 3 considered as negative")
    print()
    print('Naive Bayes: %.2f' % accuracy_score(new_y_test2, y_pred))
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
