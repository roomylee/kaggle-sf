
# coding: utf-8

# # Modify the **Dates-Minute** column to 0 if the value is 30
# 
# In previous experiment, we found that the **Dates-Minute** column has some mis-aggregation, especially 0 minute and 30 minutes. In this notebook, we'll try to ignore the data if the **Dates-Minute** value is 30. (i.e modify the value 30 to 0) And see how does it improves our model's accuracy.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("../data/train.csv")
train.head(3)


# ## Feature Engineering

# ### Convert the **Dates** column to many numerical columns

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


# ### Modify the **Dates-Minute** to 0 if the value is 30

# In[4]:

feature_names = ["X", "Y"] + ["Dates-Hour", "Dates-Minute"]

train_former_X = train[feature_names].copy()
train_former_X.head(1)


# In[10]:

train_latter_X = train[feature_names].copy()
train_latter_X.loc[(train_latter_X["Dates-Minute"] == 30), "Dates-Minute"] = 0
train_latter_X[train_former_X["Dates-Minute"] == 30].head(1)


# In[11]:

label_name = "Category"

train_y = train[label_name]
train_y.head(1)


# ## Score

# In[12]:

from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score

model = BernoulliNB()

get_ipython().magic("time former_score = cross_val_score(model, train_former_X, train_y, scoring='log_loss', cv=5).mean()")
former_score = -1.0 * former_score
print("Before change the Dates-Minute to 0 if the value is 30 w/ BernoulliNB = {0:.6f}".format(former_score))

get_ipython().magic("time latter_score = cross_val_score(model, train_latter_X, train_y, scoring='log_loss', cv=5).mean()")
latter_score = -1.0 * latter_score
score_difference = latter_score - former_score
print("After change the Dates-Minute to 0 if the value is 30 w/ BernoulliNB = {0:.6f}({1:+.6f})".format(latter_score, score_difference))


# ## Result
# ** Before change the Dates-Minute column to 0 if the value is 30 **
#   * BernoulliNB = 2.620478
# 
# ** After change the Dates-Minute column to 0 if the value is 30 **
#   * BernoulliNB = **2.589878** (-0.030600)

# In[ ]:



