
# coding: utf-8

# # Apply the 'Dates' column
# 
# In this notebook, we're going to apply the **Dates** column. We should take account the data type due to the model's limitation. (i.e our model doesn't understand datetime) Therefore, We'll try to convert the **Dates** column to many numerical columns named **Dates-Year**, **Dates-Month**, **Dates-Day**, **Dates-Hour**, **Dates-Minute**, apply to our model and see how these columns will improve our model's accuracy.

# In[26]:

import numpy as np
import pandas as pd


# ## Load Data

# In[27]:

train = pd.read_csv("../data/train.csv")
train.head(3)


# ## Feature Engineering

# ### Convert the **Dates** column to many numerical columns

# In[28]:

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

# In[35]:

former_feature_names = ["X", "Y"]

train_former_X = train[former_feature_names]
train_former_X.head(1)


# In[36]:

latter_feature_names = former_feature_names + list(dates_dataframe.columns)
train_latter_X = train[latter_feature_names]
train_latter_X.head(1)


# In[37]:

label_name = "Category"

train_y = train[label_name]
train_y.head(1)


# In[41]:

from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score

model = BernoulliNB()

get_ipython().magic("time former_score = cross_val_score(model, train_former_X, train_y, scoring='log_loss', cv=5).mean()")
print("Before applying the Dates column w/ BernoulliNB = {0:.6f}".format(-1.0 * former_score))

get_ipython().magic("time latter_score = cross_val_score(model, train_latter_X, train_y, scoring='log_loss', cv=5).mean()")
print("After applying the Dates column w/ BernoulliNB = {0:.6f}".format(-1.0 * latter_score))


# ## Result
# ** Before applying the Dates column **
#   * BernoulliNB = 2.680326
# 
# ** After applying the Dates column **
#   * BernoulliNB = **2.620481** (-0.059845)
