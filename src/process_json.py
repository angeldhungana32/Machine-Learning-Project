'''
    @author - Angel Dhunagna
    This file will read Yelp DataSet especially the reviews, and export the reviews and its star as a csv file
    Uncomment the nltk.download(), if you are running it for the first time
'''

import json
import csv
import string
import nltk
from nltk.corpus import stopwords
# If you are running this for first time, uncomment this
#nltk.download('stopwords')
#nltk.download('wordnet')
import numpy as np
import re
from nltk.stem import WordNetLemmatizer


class process_json:
    def __init__(self):
        pass

    def make_csv_from_Json(self, json_filepath, csv_file_to_make, fields,
                           max_subset, process_text_what):
        '''
            Converts the json file to csv
                json_filepath - file path and filename of json file
                csv_file_to_make - file path and name of the csv file to make
                fields - which json catergory to include in csv
                max_subset - what size of csv file
        '''
        with open(csv_file_to_make, 'w') as fout:
            count = 0
            csv_file = csv.writer(fout)
            with open(json_filepath) as fin:
                for line in fin:
                    line_contents = json.loads(line)
                    rows = []
                    for eachField in fields:
                        content = line_contents[eachField]
                        # Process each reviews or any text, more info see below
                        if process_text_what == True:
                            if eachField == 'text':
                                content = self.process_text(content)
                        rows.append(content)
                    csv_file.writerow(rows)
                    count += 1
                    if count > max_subset:
                        break

    def process_text(self, text):
        '''
            Process each text
                - Remove punctuations
                - Remove any numbers for now, just keep it to words
                - Remove Stop Words
                - Lemmatization, turn words into their root forms
                    - Going -> Go
                    - Goes -> Go
                - This process might take time, hence we can consider Lemmatization for later
                - If you wanna lemmatization just uncomment the last few codes and remove the first return
        '''
        # Remove Punctuation
        rmvpunc = [char for char in text if char not in string.punctuation]
        rmvpunc = ''.join(rmvpunc)
        # Remove any numbers or digits
        output = re.sub(r'\d+', '', rmvpunc)
        # Remove any stop words and make every word lowercase
        lower_rmstpwords = [
            word.lower() for word in output.split()
            if word.lower() not in stopwords.words('english')
        ]
        return " ".join(lower_rmstpwords)
        #lemm = WordNetLemmatizer()
        #stem = " ".join([lemm.lemmatize(i) for i in lower_rmstpwords])
        #return stem
