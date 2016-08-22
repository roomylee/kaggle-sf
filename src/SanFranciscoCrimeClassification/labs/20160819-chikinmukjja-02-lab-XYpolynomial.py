
# coding: utf-8

# # The State of the Art Script of San Francisco Crime Classification
# 
# This is a deliverable notebook of [Kaggle](https://www.kaggle.com)'s [San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime). All members who are participating this competition can apply their experiments to this notebook. The submission output should always keep the high score. When you finished to apply new experiment on this notebook, you need to review your codes using [pull request](https://help.github.com/articles/using-pull-requests/).
# 
# 
# ### Overview
# ** Model **
#   * BernoulliNB. All hyperparameters are default.
#   
# ** Features **
#   * X, Y
#       * add X*Y feature : 2.589879
#       * add X^2, Y^2 feature : 2.589881 -> 2.59008
#       
#   * Dates (Convert to numerical columns)
#     * Conver the Dates column to numerical columns named Dates-Year, Dates-Month, Dates-Day, Dates-Hour and Dates-Minute.
#     * Modify Dates-Minute to zero if the value is equal to 30.
#     * Use only Dates-Hour and Dates-Minute.
# 
# 
# ### Result
#   * 5-fold Cross Validation = **2.589878**
#   * Leaderboard = **2.59008**

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_csv("train.csv/train.csv")
train.head(3)


# In[3]:

test = pd.read_csv("test.csv/test.csv")
test.head(3)


# ## Feature Extraction

# ### Merge both the train and the test dataframe

# In[4]:

seperator = train.shape[0]

train["combi-index"] = ["train-{0}".format(index) for index, _ in train.iterrows()]
test["combi-index"] = ["test-{0}".format(index) for index, _ in test.iterrows()]

combi = pd.concat([train, test])
combi = combi.set_index("combi-index")

combi.head(3)


# ### Convert the 'Dates' column to numerical columns
# 

# In[5]:

from datetime import datetime

total_count = combi.shape[0]
count = 0

dates_data = []

for index, row in combi["Dates"].iteritems():
    count = count + 1

    if count % 100000 == 0:
        print("processing... {0}/{1}".format(count, total_count))

    date = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")

    dates_data.append({
        "combi-index": index,
        "Dates-Year": date.year,
        "Dates-Month": date.month,
        "Dates-Day": date.day,
        "Dates-Hour": date.hour,
        "Dates-Minute": date.minute,
        "Dates-Second": date.second,
    })
    
dates_dataframe = pd.DataFrame.from_dict(dates_data)
dates_dataframe = dates_dataframe.set_index("combi-index")

dates_columns = ["Dates-Year", "Dates-Month", "Dates-Day", "Dates-Hour", "Dates-Minute", "Dates-Second"]
dates_dataframe = dates_dataframe[dates_columns]

# All "Dates-Second" variable is equal to zero. Therefore, we can remove it.
second_list = dates_dataframe["Dates-Second"].unique()
print("list of seconds = {0}".format(second_list))

dates_dataframe = dates_dataframe.drop("Dates-Second", axis=1)

combi = pd.concat([combi, dates_dataframe], axis=1)

combi.head(3)


# ### Modify the **Dates-Minute** to 0 if the value is 30

# In[6]:

combi.loc[combi["Dates-Minute"] == 30, "Dates-Minute"] = 0
print("The number of rows which the Date-Minutes is equal to 30 = {0}".format(combi[combi["Dates-Minute"] == 30].shape[0]))


# ### Split to the train and the test dataframe

# In[7]:

train = combi[:seperator]
train = train.drop("Id", axis=1)
train.head(3)


# # Add X*Y feature

# In[8]:

train['XY']=train['X']*train['Y']
train['X^2'] = train['X']*train['X']
train['Y^2'] = train['Y']*train['Y']

train.head()


# In[9]:

test = combi[seperator:]
test = test.drop(["Category", "Descript", "Resolution"], axis=1)
test["Id"] = test["Id"].astype('int32')

test = test.set_index(["Id"])
test.head(3)


# In[10]:

test['XY'] = test['X']*test['Y']
test['X^2']=test['X']*test['X']
test['Y^2']=test['Y']*test['Y']

test.head(3)


# ## Score

# In[11]:

from sklearn.cross_validation import cross_val_score

feature_names = ["X", "Y","XY",'X^2','Y^2'] + ["Dates-Hour", "Dates-Minute"]
label_name = "Category"

train_X = train[feature_names]
test_X = test[feature_names]

train_y = train[label_name]

train_X.head(3)


# In[12]:

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
score = cross_val_score(model, train_X, train_y, scoring='log_loss', cv=5).mean()
print("BernoulliNB = {0:.6f}".format(-1.0 * score))


# ## Prediction

# In[13]:

model = BernoulliNB()
model.fit(train_X, train_y)

prediction = model.predict_proba(test_X)
prediction[0:100]


# # Submission

# In[26]:

sample = pd.read_csv("sampleSubmission.csv/sampleSubmission.csv", index_col="Id")
sample.head(3)


# In[27]:

submission = pd.DataFrame(prediction, index=sample.index)
submission.columns = sample.columns
submission.head()


# In[32]:

from datetime import datetime

current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
description = "Use the Dates column"

#filename = "{0} {1}.csv".format(current_time, description)

submission.to_csv("poly feature sub")


# In[ ]:



