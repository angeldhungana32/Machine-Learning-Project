from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import read_csv
import pandas as pd
import pickle
'''
    Vectorize the list of words in reviews along with their label, which in this case is rating 'Star'
    We will be using Bag of Words Process to Vectorize the words
        - This is the easiest way to vectorize words, count of words
        - Example:
            Words       Review1     Review2 ... ReviewN
            Word 1         1            0           0
            Word 2         0            2           0
            ...           ...          ...         ...
            Word N         0            1           0
        - The drawback is the it would create a large sparse matrix
        
'''


class Vectorization():
    def __init__(self):
        pass

    def vectorize(self, X, file_name):
        '''
            Read the csv file and get X and Y
                X = Processed Reviews
                Y = Rating 'Stars'
        '''
        return self.bag_of_words(X, file_name)

    def bag_of_words(self, X, file_name):
        '''
            Vectorize X by counts using Bag of Words Method
        '''
        X = CountVectorizer().fit_transform(X)
        pickle.dump(X, open(file_name, "wb"))
