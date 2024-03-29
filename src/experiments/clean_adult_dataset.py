#!/usr/bin/env python
# coding: utf-8

# This source code is taken from 
# https://www.kaggle.com/overload10/income-prediction-on-uci-adult-dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# ### Loading *csv* file in *dataframe*

# In[21]:

# downloaded from https://www.kaggle.com/overload10/adult-census-dataset
df = pd.read_csv("../../data/adult.csv", 1, ",")
data = [df]

# ### Convert *salary* to integer

# In[22]:


salary_map={' <=50K':1,' >50K':0}
df['salary']=df['salary'].map(salary_map).astype(int)

# ### convert *sex* into *integer*

df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)


# ### Drop empty value marked as '?'

df['country'] = df['country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)

df.dropna(how='any',inplace=True)

### Encode data

for dataset in data:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'
    
df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)

df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')

df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

df['marital-status'] = df['marital-status'].map({'Couple':0,'Single':1})

rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}
df['relationship'] = df['relationship'].map(rel_map)

race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}
df['race']= df['race'].map(race_map)


def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': return 'govt'
    elif x['workclass'] == ' Private':return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
df['employment_type']=df.apply(f, axis=1)
employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

df['employment_type'] = df['employment_type'].map(employment_map)
df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)


df.loc[(df['capital-gain'] > 0),'capital-gain'] = 1
df.loc[(df['capital-gain'] == 0 ,'capital-gain')]= 0


x=df['capital-loss']
plt.hist(x,bins=None)
plt.show()

df.loc[(df['capital-loss'] > 0),'capital-loss'] = 1
df.loc[(df['capital-loss'] == 0 ,'capital-loss')]= 0

df.to_csv("../../data/adults_cleaned.csv")
