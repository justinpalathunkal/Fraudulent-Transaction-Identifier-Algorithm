#importing packages
import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing packages for algorithms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print ('Python: ' + format(sys.version))
print ('Numpy: ' + format(numpy.__version__))
print ('Pandas: ' + format(pandas.__version__))
print ('Matplotlib: ' + format(matplotlib.__version__))
print ('Seaborn: ' + format(seaborn.__version__))
print ('Scipy: ' + format(scipy.__version__))


#load dataset from the Kaggle creditcard csv file
data = pd.read_csv('creditcard.csv')

#Showing the amount of fraudulent transactions and the amount of Valid Transactions 
#Finding amount of fraudulent transactions
Fraud = data[data['Class']==1]
#Finding the amount of Valid Transactions
Valid = data[data['Class']==0]

#Finding percent of Fraudulent to Valid
outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print('Fraud Transactions: {}'.format(len(Fraud)))
print('Valid Transactions: {}'.format(len(Valid)))

#Creating a matrix of correlation
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = 8)
plt.show()

#Getting columns from dataframe
columns = data.columns.tolist()

#Filtering columns with data that is not needed
columns = [c for c in columns if c not in ['Class']]

#Using and storing vairable that will be used for prediction
target = "Class"

X = data[columns]
Y = data[target]

print (X.shape)
print (Y.shape)

#define random state
state = 1

#create outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}

#Creating and Fitting the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fitting the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    
    else: 
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
       
    #Reshaping the values so 0 = Valid and 1 = Fraudulent
    
    y_pred[y_pred==1] = 0
    y_pred[y_pred==-1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

