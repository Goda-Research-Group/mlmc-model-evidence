#!/usr/bin/env python
# coding: utf-8

###### Sparse GP Classification (On adult dataset)

import tensorflow as tf
import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import datetime

import sys
sys.path.append('../')
from models.gaussian_process_classification import gaussian_process_classification


### Utility

def timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S")  


### Load Dataset

df = pd.read_csv("../../data/adults_cleaned.csv", 1, ',', index_col=0)  # if not found, run clean_adult_dataset.py
x= df.drop(['salary'],axis=1).to_numpy(dtype=np.float64)
x= (x - x.mean()) / x.std()
y=df['salary'].to_numpy()

split_size=0.3

#Creation of Train and Test dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=split_size,random_state=22)

#Creation of Train and validation dataset
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=5)


### Random Forest and Logistic Regression as a Baseline 

model = RandomForestClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)
p = model.predict_proba(x_test)[:,1]
score = (y_test*np.log(p+1e-8) + (1-y_test)*np.log(1-p+1e-8)).mean()
print()
print("Test likelihood with random forest:", score) 

model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)
p = model.predict_proba(x_test)[:,1]
score = (y_test*np.log(p) + (1-y_test)*np.log(1-p)).mean()
print("Test likelihood with logistic regression:", score)
print()


gpc = gaussian_process_classification(N_total=x_train.shape[0], M=50)
gpc.fit(x_train,y_train, learning_rate=0.005, n_iter=10001, objective='LMELBO', obj_param={'n_MC':512})
gpc.fit(x_train,y_train, learning_rate=0.001, n_iter=60001, objective='LMELBO', obj_param={'n_MC':512})
gpc.fit(x_train, y_train, learning_rate=0.0001, n_iter=20001, batch_size=2000, objective='LMELBO_MLMC', obj_param={'max_level':6, 'N0':16})
score = np.mean([gpc.score(x_test,y_test, obj_param={'n_MC':512}) for i in range(20)])
print()
print("Test likelihood with Gaussain process (LMELBO):", score)


gpc.plot_convergence(x_test[:1000], y_test[:1000], 11)
# save fig
filename = '../../out/gaussian_process_classification/dLMELBO_convergence_{}.pdf'.format(timestamp())
plt.savefig(filename)
plt.figure()
print("saved the results to:\n {}\n".format(filename))

gpc.plot_convergence_grad(x_test, y_test, 11)
# save fig
filename = '../../out/gaussian_process_classification/dLMELBO_gradients_convergence_{}.pdf'.format(timestamp())
plt.savefig(filename)
print("saved the results to:\n {}\n".format(filename))
print()

for i in range(3):
    gpc.fit(x_train,y_train, learning_rate=0.0001, batch_size=5000, n_iter=40001, objective='ELBO')
    score = np.mean([gpc.score(x_test,y_test, obj_param={'n_MC':512}) for i in range(20)])
    print()
    print("Test likelihood with Gaussain process (ELBO):", score)

    gpc.fit(x_train, y_train, learning_rate=0.0001, n_iter=40001, batch_size=3000, objective='LMELBO_MLMC', obj_param={'max_level':6, 'N0':16})
    score = np.mean([gpc.score(x_test,y_test, obj_param={'n_MC':512}) for i in range(20)])
    print()
    print("Test likelihood with Gaussain process (LMELBO):", score)
