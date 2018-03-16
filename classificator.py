from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import csv 

# Data to train the model
dataX  = pd.read_csv('./dataset/x_values_metrics.csv')
y = pd.read_csv('./dataset/y_values_metrics.csv')

# Data for Test
test_x = pd.read_csv('./dataset/x_values_test.csv')
true_y = pd.read_csv('./dataset/y_values_test.csv')


clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(dataX, np.ravel(y))

predicted_y =  clf.predict(test_x)


print(confusion_matrix(true_y, predicted_y))




