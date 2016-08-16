
# coding: utf-8

# # Baseline script of San Francisco Crime Classification

# ## Goal
#   - Make baseline script. Hope to help all mentees.
#   - SVN: 너무 오래걸려서 포기. [Link](https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution)
#   - Logistic Regression: 2.66
#   - KNN(k=500): 2.77

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("../data/train.csv")
train.head(3)


# In[3]:

test = pd.read_csv("../data/test.csv")
test.head(3)


# ## Preprocess

# In[4]:

from sklearn.utils import shuffle

train = shuffle(train, random_state=0)


# In[5]:

feature_names = ["X", "Y"]
label_name = "Category"

train_X = train[feature_names]
test_X = test[feature_names]

train_y = train[label_name]


# In[6]:

from sklearn import preprocessing

train_X = preprocessing.scale(train_X)
test_X = preprocessing.scale(test_X)


# ## Cross Validation Scoring

# In[7]:

from sklearn.cross_validation import cross_val_score


# In[ ]:

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gaussian_score = cross_val_score(GaussianNB(), train_X, train_y, scoring='log_loss', cv=5).mean()
bernoulli_score = cross_val_score(BernoulliNB(), train_X, train_y, scoring='log_loss', cv=5).mean()

print("GaussianNB = {0:.6f}".format(gaussian_score))
# print("MultinomialNB = {0:.6f}".format(multimonial_score))
print("BernoulliNB = {0:.6f}".format(bernoulli_score))


# ### Logistic Regression

# In[42]:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=-1, verbose=1)

logistic_score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=2).mean()
print("LogisticRegression = {0:.6f}".format(-1.0 * logistic_score))


# ### SVM

# In[ ]:

from sklearn import svm

model = svm.SVC(probability=True, verbose=True)
svm_score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=2).mean()
print("SVM = {0:.6f}".format(-1.0 * svm_score))


# ### KNearestNeighbors

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier

#n_neighbors를 몇으로 할지가 확실하지 않으므로 몇개 해본다 (메모리 너무 먹어서 포기)
#weights는 uniform이 좋다.

k_list = [500, 1000, 2000]
for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', n_jobs=-1)
    knn_score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=2).mean()
    print("### k={0:d}, KNearestNeighbors = {2:.6f}".format(k, -1.0 * knn_score))


# In[ ]:

#위의 결과를 보고 최적의 값으로 model을 만든다
model = KNeighborsClassifier(n_neighbors=500, weights='uniform', n_jobs=-1)


# ## Prediction

# In[10]:

model.fit(train_X, train_y)
prediction = model.predict_proba(test_X)


# ## Submission

# In[11]:

sample = pd.read_csv("../data/sampleSubmission.csv", index_col="Id")
sample.head(3)


# In[12]:

submission = pd.DataFrame(prediction, index=sample.index)
submission.columns = sample.columns
submission.head(1)


# In[13]:

from datetime import datetime

current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
description = "baseline script"

filename = "../submission/{0} {1}.csv".format(current_time, description)

submission.to_csv(filename)

