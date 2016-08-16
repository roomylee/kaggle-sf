
# coding: utf-8

# # Feature Selection using global search
# 
# In this notebook, we're going to find the optimal combination of date columns. we can't guarantee that all columns which we extracted from the **Dates** will affect positive influence of our model. In this experiment, we try to test every possible combination of dates and select features that will give us best accuracy.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("../data/train.csv")
train.head(3)


# ## Feature Engineering

# ### Convert the **Dates** column to numerical columns

# In[3]:

from datetime import datetime

total_count = train.shape[0]
count = 0

dates_data = []

for index, row in train["Dates"].iteritems():
    count = count + 1

    if count % 100000 == 0:
        print("processing... {0}/{1}".format(count, total_count))

    date = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")

    dates_data.append({
        "index": index,
        "Dates-Year": date.year,
        "Dates-Month": date.month,
        "Dates-Day": date.day,
        "Dates-Hour": date.hour,
        "Dates-Minute": date.minute,
        "Dates-Second": date.second,
    })
    
dates_dataframe = pd.DataFrame.from_dict(dates_data).astype('int32')
dates_dataframe = dates_dataframe.set_index("index")

dates_columns = ["Dates-Year", "Dates-Month", "Dates-Day", "Dates-Hour", "Dates-Minute", "Dates-Second"]
dates_dataframe = dates_dataframe[dates_columns]

# All "Dates-Second" variable is equal to zero. Therefore, we can remove it.
second_list = dates_dataframe["Dates-Second"].unique()
print("list of seconds = {0}".format(second_list))

dates_dataframe = dates_dataframe.drop("Dates-Second", axis=1)

train = pd.concat([train, dates_dataframe], axis=1)

train.head(3)


# ## Score

# In[12]:

from itertools import combinations
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cross_validation import cross_val_score

base_score = 2.620481

default_feature_names = ["X", "Y"]
default_feature_names = default_feature_names

column_length = dates_dataframe.shape[1]

combination_result = []

for i in range(column_length):
    for column_cases in combinations(dates_dataframe.columns, i+1):
        feature_names = default_feature_names + list(column_cases)
        label_name = "Category"

        train_X = train[feature_names]
        train_y = train[label_name]

        model = BernoulliNB()
        get_ipython().magic("time score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=5).mean()")
        score = -1.0 * score
        score_difference = score - base_score

        combination_text = ", ".join(column_cases)

        print("Score using \"{0}\" columns".format(combination_text))
        print("BernoulliNB = {0:.6f} ({1:+.6f})".format(score, score_difference))
        
        combination_result.append({
            'combination': combination_text,
            'model': "BernoulliNB",
            "score": score,
        })
        
combination_result_dataframe = pd.DataFrame.from_dict(combination_result)
combination_result_dataframe = combination_result_dataframe.set_index("combination")

combination_result_dataframe = combination_result_dataframe.sort("score")
combination_result_dataframe.head(5)


# ## Result
#   * default = 2.620481
# 
# 
#   * Select only Dates-Hour and Dates-Minute = **2.620478** (-0.000003)
#   * Select only Dates-Month, Dates-Hour and Dates-Minute = 2.620479 (-0.0000022)
#   * Select only Dates-Year, Dates-Hour and Dates-Minute = 2.620479 (-0.0000022)
#   * Select only Dates-Day, Dates-Hour and Dates-Minute = 2.620479 (-0.000002)
#   * Select only Dates-Year, Dates-Day, Dates-Hour and Dates-Minute = 2.620480 (-0.000001)

# In[ ]:



