#!/usr/bin/python


#   This gets approximately 95% accuracy, equivelent to my R attempt
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

# Normalizing data
x_train = (x_train / (255*2)) - 1      
x_valid = (x_valid / (255*2)) - 1
print 'normalized data !'

classifier = SVC(kernel="rbf", C=2.8, gamma=.0073)
print 'instigated classifier model !'
y_pred = classifier.fit(x_train, y_train).predict(x_valid)
print 'building based on training data set !'

cm = confusion_matrix(y_valid, y_pred)
asm = accuracy_score(y_valid,y_pred)
csm = classification_report(y_valid,y_pred)
print(cm)
print(asm) #Shows ~95% accuracy which is ok
print(csm)



