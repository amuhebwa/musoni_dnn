# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:51:53 2016

@author: aggrey
"""
import pandas as pd

import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import skflow
import random

random.seed(42)
#Load training data
df_train = pd.read_csv('musoni/groups_final_dataset.csv', low_memory=False)
train_data = df_train.drop(['Average Over Due Days'], axis = 1)
train_data = train_data.sample(frac = 1).reset_index(drop=True)
x_train = df_train[['Number Of Loans','Average Principal','Past One Month', 'Past Three Months']]
y_train = df_train[['Low/High Risk']]

# Load testing data
df_test = pd.read_csv('musoni/test.csv', low_memory=False)
x_test = df_test[['Number Of Loans','Average Principal','Past One Month', 'Past Three Months']]
y_test = df_test[['Low/High Risk']]


# setup exponential decay function
def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=0.01, global_step=global_step,
        decay_steps=100, decay_rate=0.001)


# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10,20, 50, 20, 10],n_classes=2, steps=100000,learning_rate=exp_decay)

# Fit and predict.
classifier.fit(x_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(x_test))
print('Accuracy: {0:f}'.format(score))

