
# coding: utf-8

# # Baseline script of San Francisco Crime Classification

# ## Goal
#   * Make baseline script. Hope to help all mentees.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("../data/train.csv")
train = train.drop(['Dates', 'Descript', 'Resolution'], axis=1)
train.head(3)


# In[3]:

test = pd.read_csv("../data/test.csv")
test.head(3)


# ## Matching Sanfran Hot Place
#  - PIER 39 : (37.808690, -122.409831)
#  - Golden Gate Park : (37.968462, -122.480034)
#  - Union square : (37.787951, -122.407541)
#  - Golden Gate Bridge : (37.808978, -122.475337)

# In[4]:

hot_place = {'pier': {'x': -122.409831, 'y': 37.808690},
             'park': {'x': -122.480034, 'y': 37.968462}, 
             'union': {'x': -122.407541, 'y': 37.787951},
             'bridge': {'x': -122.475337, 'y': 37.808978}
            }


# In[5]:

hot_place['pier'].get('x')


# In[6]:

# man_dist = ((dest.x - curr.x) + (dest.y - curr.y)) 

def getManhattan(xList, yList):
    return abs(xList[0]-xList[1]) + abs(yList[0]-yList[1])


# ## Feature Engineering
# - 각 거리를 계산하여 4개의 feature를 생성
# - 소수점 자리가 너무 낮은 관계로 100을 곱함

# In[7]:

xList = []
yList = []


# In[8]:

train['PierCalcX'] = train['X'].map(lambda x: abs(hot_place['pier'].get('x') - x))
train['PierCalcY'] = train['Y'].map(lambda y: abs(hot_place['pier'].get('y') - y))
train['PierCalc'] = (train['PierCalcX'] + train['PierCalcY']) * 100

train['ParkCalcX'] = train['X'].map(lambda x: abs(hot_place['park'].get('x') - x))
train['ParkCalcY'] = train['Y'].map(lambda y: abs(hot_place['park'].get('y') - y))
train['ParkCalc'] = (train['ParkCalcX'] + train['ParkCalcY']) * 100

train['UnionCalcX'] = train['X'].map(lambda x: abs(hot_place['union'].get('x') - x))
train['UnionCalcY'] = train['Y'].map(lambda y: abs(hot_place['union'].get('y') - y))
train['UnionCalc'] = (train['UnionCalcX'] + train['UnionCalcY']) * 100

train['BridgeCalcX'] = train['X'].map(lambda x: abs(hot_place['bridge'].get('x') - x))
train['BridgeCalcY'] = train['Y'].map(lambda y: abs(hot_place['bridge'].get('y') - y))
train['BridgeCalc'] = (train['BridgeCalcX'] + train['BridgeCalcY']) * 100

train = train.drop(['PierCalcX', 'PierCalcY', 'ParkCalcX', 'ParkCalcY', 'UnionCalcX', 'UnionCalcY', 'BridgeCalcX', 'BridgeCalcY'], axis=1)
train.head(3)


# ## Prediction

# In[11]:

from sklearn.cross_validation import cross_val_score

feature_names = ["X", "Y", "PierCalc", "ParkCalc", "UnionCalc", "BridgeCalc"]
label_name = "Category"

train_X = train[feature_names]
train_y = train[label_name]


# ### Scoring

# In[12]:

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gaussian_score = cross_val_score(GaussianNB(), train_X, train_y, scoring='log_loss', cv=5).mean()
bernoulli_score = cross_val_score(BernoulliNB(), train_X, train_y, scoring='log_loss', cv=5).mean()

print("GaussianNB = {0:.6f}".format(-1.0 * gaussian_score))
# print("MultinomialNB = {0:.6f}".format(multimonial_score))
print("BernoulliNB = {0:.6f}".format(-1.0 * bernoulli_score))


# ### Result
# - baseline : GaussianNB = 3.456489 / BernoulliNB = 2.680326
# - hotplace : GaussianNB = 8.392301 / BernoulliNB = 2.680327 (+0.000001)
# - hotplace : GaussianNB = 10.799314 / BernoulliNB = 2.680330 (+0.000004)
