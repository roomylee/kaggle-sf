
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps yours.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("train.csv")
train.head(3)


# In[3]:

test = pd.read_csv("test.csv")
test.head(3)


# In[4]:

from sklearn.cross_validation import cross_val_score

feature_names = ["X", "Y"]
label_name = "Category"

train_X = train[feature_names]
test_X = test[feature_names]

train_y = train[label_name]


# ## Score

# In[5]:

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=5).mean()
print("BernoulliNB = {0:.6f}".format(-1.0 * score))


# ## Prediction

# In[6]:

model = BernoulliNB()
model.fit(train_X, train_y)

prediction = model.predict_proba(test_X)
prediction[0:100]


# # Submission

# In[7]:

sample = pd.read_csv("sampleSubmission.csv", index_col="Id")
sample.head(3)


# In[8]:

submission = pd.DataFrame(prediction, index=sample.index)
submission.columns = sample.columns
submission.head()


# In[9]:

from datetime import datetime

current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
description = "baseline script"

filename = "{0} {1}.csv".format(current_time, description)

submission.to_csv(filename)

