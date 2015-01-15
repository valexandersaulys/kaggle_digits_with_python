#!/usr/bin/python2


#   Attempting the MNIST train.csv set
#   Using Python's scikit svm kernels
#   Attempting to beat my 95.5% record on Kaggle as renfieldsdrunk
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Completed with ample help from the internets


#from pandas import read_csv
from sklearn import cross_validation
from sklearn.svm import LinearSVC
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

#mm_scal = MinMaxScaler()
#pca = PCA(n_components=99)

#x_train = mm_scal.fit_transform(x_train)
#x_train = pca.fit_transform(x_train)
#x_valid = mm_scal.fit_transform(x_valid)
#x_valid = pca.fit_transform(x_valid)

                    
print 'everything all set, preparing our model!'                    
linear_svm = LinearSVC(penalty='l2', 
						loss='l2', 
						dual=True,		# Prefer dual=False when n_samples > n_features.
						tol=0.0001, 
						C=1.0, 
						multi_class='ovr', 
						fit_intercept=True,
						intercept_scaling=1, 
						class_weight=None, 
						verbose=0, 
						random_state=None
						)
fitted_model = linear_svm.fit(x_train, y_train)
print 'preparation done!'

# Fit to some predictors
y_pred = fitted_model.predict(x_valid)

# Print our output!
cm = confusion_matrix(y_valid, y_pred)
asm = accuracy_score(y_valid,y_pred)
print(cm) #ConfusionMatrix
print "Accuracy: %f" % (asm) #accuracy

"""
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
""" 
