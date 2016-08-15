
# coding: utf-8

# # Baseline script of San Francisco Crime Classification

# ## Goal
#   * Make baseline script. Hope to help all mentees.

# In[4]:

import numpy as np
import pandas as pd


# ## Load Data

# In[5]:

train = pd.read_csv("../data/train.csv")
train.head(3)


# In[6]:

test = pd.read_csv("../data/test.csv")
test.head(3)


# ## Munging

# In[38]:

train['AddrSt'] = train['Address'].map(lambda x: 1 if 'ST' in x else 0)
train['AddrAv'] = train['Address'].map(lambda x: 1 if 'AV' in x else 0)
train['AddrDr'] = train['Address'].map(lambda x: 1 if 'DR' in x else 0)

test['AddrSt'] = test['Address'].map(lambda x: 1 if 'ST' in x else 0)
test['AddrAv'] = test['Address'].map(lambda x: 1 if 'AV' in x else 0)
test['AddrDr'] = test['Address'].map(lambda x: 1 if 'DR' in x else 0)


# ## Prediction

# In[44]:

from sklearn.cross_validation import cross_val_score

feature_names = ["X", "Y", 'AddrSt', 'AddrAv', 'AddrDr']
label_name = "Category"

train_X = train[feature_names]
test_X = test[feature_names]
train_y = train[label_name]


# ### Scoring

# In[45]:

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gaussian_score = cross_val_score(GaussianNB(), train_X, train_y, scoring='log_loss', cv=5).mean()
bernoulli_score = cross_val_score(BernoulliNB(), train_X, train_y, scoring='log_loss', cv=5).mean()

print("GaussianNB = {0:.6f}".format(gaussian_score))
# print("MultinomialNB = {0:.6f}".format(multimonial_score))
print("BernoulliNB = {0:.6f}".format(bernoulli_score))

# baseline
# GaussianNB = -3.456489
# BernoulliNB = -2.680326


# In[ ]:



