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
import os

import pandas as pd


# Polyaxon
experiment = Experiment()


dataset = datasets.load_boston()
# x 训练特征：['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
#'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = dataset.data
 
target = dataset.target
#把label变为(?, 1)维度，为了使用下面的数据集合分割
y = np.reshape(target,(len(target), 1))
 
#讲数据集1:3比例分割为 测试集：训练集
x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)

'''
x_train的shape：(379, 13)
y_train的shape：(379, 1)
x_verify的shape：(127, 13)
y_verify 的shape：(127, 1)
'''
# Polyaxon
experiment.log_data_ref(data=x_train, data_name='dataset_X')
experiment.log_data_ref(data=y_train, data_name='dataset_y')



 
'''----------定义线性回归模型，进行训练、预测-----------'''
lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_verify)
 
 
'''----------输出模型参数、评价模型-----------'''
print(lr.coef_)
print(lr.intercept_)
print("MSE:",metrics.mean_squared_error(y_verify,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_verify,y_pred)))


#输出模型对应R-Square
R2_score_train = lr.score(x_train,y_train)
R2_score_verify = lr.score(x_verify,y_verify)


print('R2-socre train: {}; R2-score val :{}'.format(R2_score_train, R2_score_verify))
# Polyaxon
#这个参数可以自定义吗
experiment.log_metrics(R2_score_train=R2_score_train, R2_score_val=R2_score_verify)
