#!/usr/bin/env python
# coding: utf-8

# This scripts computes the baseline predictive log likelihood for experiments using random effect logistic regression.

###### Sparse GP Classification (On adult dataset)

import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib import pyplot as plt
import datetime
import optuna

import sys
sys.path.append('../')
# from models.gaussian_process_classification import gaussian_process_classification


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

# Gaussian Process Classifier
model = GaussianProcessClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)
p = model.predict_proba(x_test)[:,1]
score = (y_test*np.log(p+1e-8) + (1-y_test)*np.log(1-p+1e-8)).mean()

print()
print("Test likelihood with Gausssian Process Classifier (Sklearn):", score) 

### Random Forest and Logistic Regression as a Baseline 
def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }
    model = RandomForestClassifier(random_state=0, **params)
    model.fit(x_train, y_train)
    model.score(x_test, y_test)
    p = model.predict_proba(x_test)[:,1]
    score = (y_test*np.log(p+1e-8) + (1-y_test)*np.log(1-p+1e-8)).mean()
    return score

study = optuna.create_study()
study.optimize(objective, n_trials=50)

params = study.best_params
model = RandomForestClassifier(random_state=0, **params)
model.fit(x_train, y_train)
model.score(x_test, y_test)
p = model.predict_proba(x_test)[:,1]
score = (y_test*np.log(p+1e-8) + (1-y_test)*np.log(1-p+1e-8)).mean()

print()
print("Test likelihood with Random Forest:", score) 
