# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:56:18 2018

@author: Vaibhav Murkute
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile

dataset = pd.read_csv("NBAstats.csv")
x = dataset.iloc[:,4:25].values
y = dataset.iloc[:,1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=1)

select1 = SelectPercentile(percentile=80)
select1.fit(x_train, y_train)
x_train_selected = select1.transform(x_train)

select2 = SelectPercentile(percentile=80)
select2.fit(x_test, y_test)
x_test_selected = select2.transform(x_test)


clf = svm.SVC(gamma=0.00007, C=1500)
clf.fit(x_train_selected,y_train)

y_predictions = clf.predict(x_test_selected)

print("SVM predictions:\n{}".format(y_predictions))
print ("\nTest set accuracy: {}".format(metrics.accuracy_score(y_test, y_predictions)))
print ("\nConfusion Matrix: \n")
print(pd.crosstab(y_test, y_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

#================ Cross Validation ====================

print("\n")
cv_accuracy = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
print("Fold \t Accuracy")
for i in range(len(cv_accuracy)):
    print("\n{} \t {}".format((i+1),cv_accuracy[i]))

print("\n")
print("Average Accuracy : {}".format(cv_accuracy.mean()))
