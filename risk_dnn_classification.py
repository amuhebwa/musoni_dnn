# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:41:14 2016

@author: aggrey
"""

import tensorflow as tf
import pandas as pd
from tensorflow.contrib import skflow
from sklearn import metrics, cross_validation

#Load training data
dataset = pd.read_csv('musoni/groups_final_dataset.csv', low_memory=False)

target = dataset['Low/High Risk']
data =  dataset.drop(['Average Over Due Days', 'Low/High Risk'], axis = 1)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, target,test_size=0.2,
                                                                     random_state=30)

# setup exponential decay function
def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=0.1, global_step=global_step,
        decay_steps=100, decay_rate=0.001)

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 40, 40, 20, 10],n_classes=2, steps=20000, learning_rate=exp_decay)
classifier.fit(x_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(x_test))
print('Accuracy: {0:f}'.format(score))
#print(classifier.predict_proba(x_test))
