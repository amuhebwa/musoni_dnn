# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:10:57 2016

@author: aggrey
"""

import tensorflow as tf
import pandas as pd
from tensorflow.contrib import skflow
import random
from sklearn import metrics, cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#Load training data
dataset = pd.read_csv('musoni/groups_final_dataset.csv', low_memory=False)
target = dataset['Low/High Risk']
data =  dataset.drop(['Average Over Due Days', 'Low/High Risk'], axis = 1)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, target,test_size=0.2,random_state=30)

"""
 1. Display the importance of each of the features
 """
#model = ExtraTreesClassifier()
#model.fit(data, target)
# display the relative importance of each attribute
#print((model.feature_importances_)*100)
#print(data.columns)

"""
2 . Choose the best set of features that produce the maximum impact

"""
#model = LogisticRegression()
#rfe = RFE(model, 1)
#rfe.fit(data, target)
#Summerize the selection of attributes
#print(rfe.support_)
#print(rfe.ranking_)
#print(data.columns)


model = LogisticRegression()
model.fit(x_train, y_train)
expected = y_test
predicted = model.predict(x_test)
probability = model.predict_proba(x_test)
classification_report = metrics.classification_report(expected, predicted)
# summarize the fit of the model
print(probability * 100)
