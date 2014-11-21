classifier = SVC(kernel="rbf", C=2.8, gamma=.0073)
print 'instigated classifier model !'
y_pred = classifier.fit(x_train, y_train).predict(x_valid)
print 'building based on training data set !'
scores = cross_validation.cross_val_score(classifier, x_train, y_train, cv=7)
vcores = cross_validation.cross_val_score(classifier, x_, cv=7)
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
print "validation set accuracy:  %0.2f (+/- %0.2f)" % ()



#cm = confusion_matrix(y_valid, y_pred)
#asm = accuracy_score(y_valid,y_pred)
#csm = classification_report(y_valid,y_pred)
#print(cm)
#print(asm)
#print(csm)