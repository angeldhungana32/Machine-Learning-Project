def read_csv_file(file_name):
    '''
        Read the csv file and get X and Y
    '''
    X = []
    Y = []
    with open(file_name) as f:
        for row in f:
            single_review = row.strip().split(",")
            X.append(single_review[0].strip())
            Y.append(single_review[1].strip())
    return X, Y
