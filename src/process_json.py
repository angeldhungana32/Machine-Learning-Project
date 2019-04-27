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
'''
    This file will read Yelp DataSet and gets the needed data into CSV file
    Nothing needs to done here, all these functions are helper Functions.
   
    Uncomment the nltk.download(), if you are running it for the first time
'''


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
        with open(csv_file_to_make, 'w', newline='') as fout:
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

    def make_metadata_from_user_reviews(self, json_filepath, csv_file_to_make,
                                        fields, num_of_users):
        '''
            Reads User data, get the top users by their review count
                json_filepath - file path and filename of json file
                csv_file_to_make - file path and name of the csv file to make
                fields - which json catergory to include in csv, user_id, name and review count
                num_of_users - how many top users to get 
        '''
        top_users = {}
        with open(csv_file_to_make, 'w', newline='') as fout:
            csv_file = csv.writer(fout)
            with open(json_filepath) as fin:
                for line in fin:
                    line_contents = json.loads(line)
                    rows = []
                    count = 0
                    for eachField in fields:
                        if eachField == 'review_count':
                            count = int(line_contents[eachField])
                        content = line_contents[eachField]
                        rows.append(content)
                    if len(top_users) < num_of_users:
                        top_users[count] = rows
                    else:
                        key_min = min(
                            top_users.keys(), key=(lambda k: top_users[k]))
                        if key_min < count:
                            del top_users[key_min]
                            top_users[count] = rows
            for key in reversed(sorted(top_users.keys())):
                csv_file.writerow(top_users[key])

    def make_csv_for_top_users_from_Json(self, json_filepath, meta_data_file,
                                         fields, process_text_what):
        '''
            Converts the json file to csv
                json_filepath - file path and filename of json file, reviews
                meta_data_file - file path and name of the metadata file
                fields - which json catergory to include in csv
                max_subset - what size of csv file
        '''
        top_reviewers = {}
        # Get the metadata and put it in dictionary
        with open(meta_data_file) as f:
            readCSV = csv.reader(f, delimiter=',')
            for each_row in readCSV:
                top_reviewers[each_row[0]] = each_row
        name = "top_reviewers/topreviewers.csv"
        fout = open(name, 'w', newline='')
        csv_file = csv.writer(fout)
        count = 0
        with open(json_filepath) as fin:
            for line in fin:
                line_contents = json.loads(line)
                rows = []
                idx = line_contents["user_id"]
                if idx in top_reviewers:
                    for eachField in fields:
                        content = line_contents[eachField]
                        # Process each reviews or any text, more info see below
                        if process_text_what == True:
                            if eachField == 'text':
                                content = self.process_text(content)
                        rows.append(content)
                    count += 1
                    csv_file.writerow(rows)
                    print("Processed " + str(count) + " reviews.")

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
                - If you want lemmatization just uncomment the last few codes and remove the first return
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
