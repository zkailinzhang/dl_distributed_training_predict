# -*- coding:utf-8 -*-

import argparse
import numpy as np

# Polyaxon
from polyaxon_client.tracking import Experiment

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
import os

import pandas as pd


# 
experiment = Experiment()

X,y = datasets.make_classification(n_features=4,random_state=0)
# 
experiment.log_data_ref(data=X, data_name='dataset_X')
experiment.log_data_ref(data=y, data_name='dataset_y')

print('data:{},lable:{}'.format(X,y))


#调用svm classification API
clf = LinearSVC(random_state=0, tol=1e-5)


clf.fit(X, y)

precision = clf.score(X,y)

print('svm score:{},'.format(clf.score(X,y)))


experiment.log_metrics(precision=precision)
