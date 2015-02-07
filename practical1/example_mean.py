import csv
import gzip
import numpy as np
from sklearn import linear_model

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = '1_naive_regression.csv'

# Load the training file.
train_data = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    temp = 0
    for row in train_csv:
        temp += 1
        if temp % 10000 == 0:
            print temp
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
    print train_data[0]

# Compute the mean of the gaps in the training data.
#gaps = np.array([datum['gap'] for datum in train_data])
#mean_gap = np.mean(gaps)

# linear regression
X = np.vstack(tuple([d['features'] for d in train_data]))
print X.shape
y = [d['gap'] for d in train_data]
clf = linear_model.LinearRegression()
clf.fit(X, y)
print clf.coef_

# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })

# Write a prediction file.
with open(pred_filename, 'wb') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for datum in test_data:
        #pred_csv.writerow([datum['id'], mean_gap])
        pred_csv.writerow([datum['id'], clf.predict(test_data['features'])])
    print "done"