from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
'''
   Runs the Bagging Classifier
'''


class _Bagging:
    def __init__(self, file_name, X_train, y_train, X_test, y_test):
        self.f = open(file_name, 'w')
        self.Bagging_Classifier_3_positive(X_train, y_train, X_test, y_test)
        self.Bagging_Classifier_3_negative(X_train, y_train, X_test, y_test)

    def Bagging_Classifier_3_positive(self, X_train, y_train, X_test, y_test):
        '''
            Make binary classification of positive or negative
            Considering star rating 3 as positive
            BaggingClassifier
        '''
        clf = BaggingClassifier(
            KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        # Change the ratings to
        new_y_train1 = self.process_Y1(y_train)
        new_y_test1 = self.process_Y1(y_test)
        clf.fit(X_train, new_y_train1)
        y_pred = clf.predict(X_test)
        self.write_to_file("\n")
        self.write_to_file("Star Rating 3 considered as positive")
        self.write_to_file("\n")
        self.write_to_file(
            'Baggin Accuracy: %.2f' % accuracy_score(new_y_test1, y_pred))
        self.write_to_file("\n")
        self.write_to_file('Classification Report:')
        self.write_to_file(classification_report(new_y_test1, y_pred))

    def process_Y1(self, y_):
        '''
            Considering 3 as positive
        '''
        newY = []
        for i in range(len(y_)):
            if float(y_[i]) < 3.0: newY.append(-1)
            else: newY.append(1)
        return newY

    def Bagging_Classifier_3_negative(self, X_train, y_train, X_test, y_test):
        '''
            Make binary classification of positive or negative
            Considering star rating 3 as positive
            Run the Bagging Classifier and Print Accuracy
        '''
        clf = BaggingClassifier(
            KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        new_y_train2 = self.process_Y2(y_train)
        new_y_test2 = self.process_Y2(y_test)
        clf.fit(X_train, new_y_train2)
        y_pred = clf.predict(X_test)
        self.write_to_file("\n")
        self.write_to_file("Star Rating 3 considered as negative")
        self.write_to_file("\n")
        self.write_to_file(
            'Bagging Accuracy: %.2f' % accuracy_score(new_y_test2, y_pred))
        self.write_to_file("\n")
        self.write_to_file('Classification Report:')
        self.write_to_file(classification_report(new_y_test2, y_pred))

    def process_Y2(self, y_):
        '''
        Considering 3 as negative
        '''
        newY = []
        for i in range(len(y_)):
            if float(y_[i]) <= 3.0: newY.append(-1)
            else: newY.append(1)
        return newY

    def write_to_file(self, text):
        self.f.writelines(text)