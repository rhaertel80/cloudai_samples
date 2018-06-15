"""Python script to train sklearn model using the criteo dataset."""

import argparse
import csv
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
import subprocess
import os

from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tensorflow.python.lib.io import file_io

import xgboost

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', default='.')
parser.add_argument('--event-date',
                    default=(date.today() - timedelta(1)).strftime('%Y%m%d'))
args = parser.parse_args()


# Dataset: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

# Number of data samples to use for training.
MAX_SAMPLES = 70000


# Read in 7 days worth of data (starting yesterday) into a key value format.

# NB: end_date is exclusive
def daterange(end_date, num_days):
    for n in range(num_days):
        yield end_date - timedelta(num_days - n)

# Data in key value format. Each element is a dict.
data = []

# Labels for the data.
y = []

num_samples = 0
for d in daterange(datetime.strptime(args.event_date, '%Y%m%d'), 7):
  data_fn = os.path.join(args.base_dir, 'logs', d.strftime('%Y%m%d'), 'click.txt')
  print('Reading file {}'.format(data_fn))
  with file_io.FileIO(data_fn, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    while num_samples < MAX_SAMPLES:
      line = reader.next()
      row = {}
      y.append(int(line[0]))
      for i in range(1, 13 + 1):
        if line[i]:
          row[str(i)] = int(line[i])
      for i in range(14, 39 + 1):
        if line[i]:
          row[str(i)] = line[i]
      data.append(row)
      num_samples += 1

# Split data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print 'training data size: {}'.format(len(X_train))
print 'test data size: {}'.format(len(X_test))

param = {
    'max_depth': 7,
    'eta': 0.2,
    'silent': False,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'gamma': 0.4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 20,
    'alpha': 3,
    'lambda': 100
}
xgb = xgboost.XGBRegressor(n_estimators=2, **param)

# Setup and train the pipeline.
pipeline = Pipeline(steps=[("dict_vect", DictVectorizer()),
                           ("xgboost", xgb)])
pipeline.fit(X_train, y_train)

# Export the model.
model = 'model.joblib'
joblib.dump(pipeline, model)

# Copy the model to its final destination
model_path = os.path.join(args.base_dir, 'models', args.event_date, model)
subprocess.check_call(['gsutil', 'cp', model, model_path])
