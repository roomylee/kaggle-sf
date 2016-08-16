
# coding: utf-8

# Learning curve of first baseline (X, Y)
# ===

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps yours.

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd


# ## Load Data

# In[3]:

train = pd.read_csv("train.csv")
train.head(3)


# In[4]:

test = pd.read_csv("test.csv")
test.head(3)


# In[5]:

from sklearn.cross_validation import cross_val_score

feature_names = ["X", "Y"]
label_name = "Category"

train_X = train[feature_names]
test_X = test[feature_names]

train_y = train[label_name]


# ## Model

# In[6]:

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()


# ## Drawing Plot
# Source: [Scikit-learn: Plot Learning Curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

# In[7]:

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[8]:

plot_learning_curve(model, "Learning curve plot of Baseline (X, Y)", train_X, train_y, cv=5, train_sizes=np.array([0.0001, 0.0002, 0.775, 1.]))
# plt.show()
plt.savefig('Learning_curve-X,Y.png')


# ## Result
# - CV 그래프가 일자
# - 16.8.16 스터디: feature가 X, Y뿐이라 CV 그래프가 일자인 것으로 예상
