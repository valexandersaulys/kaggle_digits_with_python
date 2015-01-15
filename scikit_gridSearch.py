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

tune_grid = [{'kernel' : ['rbf'], 'gamma': [0.0073], 
				'C': [0.1, 1, 2, 3] }]	# the 95.5% was achieved with C=2, gamma=0.0073
                    
best_model = GridSearchCV( SVC(), tune_grid, cv=10, verbose=2, n_jobs=8).fit(x_train, y_train)

# Give Best Estimators
BE = best_model.best_estimator_ 
print(BE)

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

y_submit = best_model.predict(df_submit)

print "All Predicted! Now writing to csv..."
pd.DataFrame(y_submit).to_csv("submittion_122614_dos.csv", index=True)
print "Complete!"
