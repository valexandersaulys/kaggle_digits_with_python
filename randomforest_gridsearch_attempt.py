#!/usr/bin/pythonprint 

#   Attempting the MNIST train.csv set
#   Using Python's scikit randomForest
#   Attempting to beat my 95.5% record on Kaggle as renfieldsdrunk
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Completed with ample help from the internets


#	Results
#	* * *	Gets about ~94% on the defaults
#	* * *	

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

classifier = RandomForestClassifier(n_jobs=4, 
									n_estimators=10,
									criterion='gini'
									)
print 'instigated classifier model !'
y_pred = classifier.fit(x_train, y_train).predict(x_valid)
print 'building based on training data set !'

cm = confusion_matrix(y_valid, y_pred)
asm = accuracy_score(y_valid,y_pred)
csm = classification_report(y_valid,y_pred)
print(cm)
print(asm) 
print(csm)

print "Now onto the Test Data!"
df_submit = pd.read_csv('test.csv')
df_submit = df_submit.astype(float)
print "Data Read..."

y_submit = classifier.predict(df_submit)

print "All Predicted! Now writing to csv..."
pd.DataFrame(y_submit).to_csv("submittion_122614_uno.csv", index=True)
print "Complete!"
