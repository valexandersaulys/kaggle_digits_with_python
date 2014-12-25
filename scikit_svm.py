#!/usr/bin/python


#   Attempting the MNIST train.csv set
#   Using Python's scikit svm kernels
#   Attempting to beat my 95.5% record on Kaggle as renfieldsdrunk
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Completed with ample help from the internets


#from pandas import read_csv
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# To reset to use all cores
import os
os.system("taskset -p 0xff %d" % os.getpid())

print 'starting...'

df = pd.read_csv('train.csv')
df = df.astype(float)
print 'read data !'

training, validation = df[:31500], df[31500:]
training = numpy.asarray(training)
validation = numpy.asarray(validation)
print 'partitioned data !'

# Splitting Training Up
y_train = training[:,0]    # labels
x_train = training[:,1:]   # everything else
# Splitting validation up
y_valid = validation[:,0]
x_valid = validation[:,1:]
print 'did necesary preparations !'

mm_scal = MinMaxScaler()
pca = PCA(n_components=99)

x_train = mm_scal.fit_transform(x_train)
x_train = pca.fit_transform(x_train)
x_valid = mm_scal.fit_transform(x_valid)
x_valid = pca.fit_transform(x_valid)

tune_grid = [{'kernel' : ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'C': [1, 10, 100]}, 
                    {'kernel' : ['poly'], 'degree' : [3, 5, 7, 9], 'C' : [1, 5, 10, 100]}]
                    
print 'everything all set, preparing the different models'                    
best_model = GridSearchCV( SVC(), tune_grid, cv=10, verbose=2, n_jobs=8).fit(x_train, y_train)

# Give Best Estimators
best_model.best_estimator_ 

# Fit to some predictors
y_pred = best_model.predict(x_valid)

# Print our output!
cm = confusion_matrix(y_valid, y_pred)
asm = accuracy_score(y_valid,y_pred)
print(cm) #ConfusionMatrix
print "Accuracy: %f" % (asm) #accuracy

print "Now onto the Test Data!"
df_submit = pd.read_csv('test.csv')
df_submit = df_submit.astype(float)
print "Data Read..."

x_submit = mm_scal.fit_transform(df_submit)
x_submit = pca.fit_transform(x_submit)

y_submit = best_model.predict(x_submit)

print "All Predicted! Now writing to csv..."
DataFrame(y_submit).to_csv("submittion_122414.csv", sep='\t', index=True)
print "Complete!"
