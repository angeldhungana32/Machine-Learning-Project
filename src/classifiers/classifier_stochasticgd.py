from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
'''
   Runs the Stochastic Gradient
'''


class Stochastic_Gradient_Descent:
    def __init__(self, file_name, X_train, y_train, X_test, y_test):
        self.f = open(file_name, 'w')
        self.SGD_Classifier_3_positive(X_train, y_train, X_test, y_test)
        self.SGD_Classifier_3_negative(X_train, y_train, X_test, y_test)

    def SGD_Classifier_3_positive(self, X_train, y_train, X_test, y_test):
        '''
            Make binary classification of positive or negative
            Considering star rating 3 as positive
            Stochastic Gradient Classifier
        '''
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
        # Change the ratings to
        new_y_train1 = self.process_Y1(y_train)
        new_y_test1 = self.process_Y1(y_test)
        clf.fit(X_train, new_y_train1)
        y_pred = clf.predict(X_test)
        self.write_to_file("\n")
        self.write_to_file("Star Rating 3 considered as positive")
        self.write_to_file("\n")
        self.write_to_file('Stochastic Gradient Descent Accuracy: %.2f' %
                           accuracy_score(new_y_test1, y_pred))
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

    def SGD_Classifier_3_negative(self, X_train, y_train, X_test, y_test):
        '''
            Make binary classification of positive or negative
            Considering star rating 3 as positive
            Run the Stochastic Gradient Classifier and Print Accuracy
        '''
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
        new_y_train2 = self.process_Y2(y_train)
        new_y_test2 = self.process_Y2(y_test)
        clf.fit(X_train, new_y_train2)
        y_pred = clf.predict(X_test)
        self.write_to_file("\n")
        self.write_to_file("Star Rating 3 considered as negative")
        self.write_to_file("\n")
        self.write_to_file('Stochastic Gradient Descent Accuracy: %.2f' %
                           accuracy_score(new_y_test2, y_pred))
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