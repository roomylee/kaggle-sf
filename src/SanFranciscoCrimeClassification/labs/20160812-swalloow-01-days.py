
# coding: utf-8

# # Baseline script of San Francisco Crime Classification

# ## Goal
#   * All about Dates.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[96]:

train = pd.read_csv("../data/train.csv")
train = train.drop(['Descript', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'], axis=1)
train.head(3)


# In[3]:

test = pd.read_csv("../data/test.csv")
test.head(3)


# ## Data Munging

# In[4]:

from datetime import datetime


# In[97]:

train['Year'] = pd.to_datetime(train.Dates).dt.year
train['Month'] = pd.to_datetime(train.Dates).dt.month
train['Days'] = pd.to_datetime(train.Dates).dt.day
train['Hour'] = pd.to_datetime(train.Dates).dt.hour
train['Minute'] = pd.to_datetime(train.Dates).dt.minute
train['Second'] = pd.to_datetime(train.Dates).dt.second
train.head()


# ## Data Analysis - Crimes by Year

# In[16]:

df = train[['Category','Year']].groupby(['Year']).count()
df.plot(y='Category', label='Number of events', figsize=(10,5)) 
plt.title("Distribution of Crimes by Year")
plt.ylabel('Number of crimes')
plt.xlabel('Year')
plt.grid(True)

plt.savefig('Distribution_of_Crimes_by_Year.png')


# ![title](Distribution_of_Crimes_by_Year.png)

# ## Data Analysis - Crimes by Hour

# In[6]:

import matplotlib.pyplot as plt


# In[7]:

month_dict={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
data_month_hour = pd.crosstab(train['Hour'],train['Month'])
axhandles = data_month_hour.plot(kind='bar',subplots=True,layout=(4,3),figsize=(16,12),sharex=True,sharey=True,xticks=range(0,24,4),rot=0)
i=1
for axrow in axhandles:
    for ax in axrow:
        ax.set_xticklabels(range(0,24,1))
        ax.legend([month_dict[i]],loc='best')
        ax.set_title("")
        i+=1
plt.suptitle('Distribution of Crimes by Hour',size=20)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('Distribution_of_Crimes_by_Hour.png')


# ![title](Distribution_of_Crimes_by_Hour.png)

# ## Data Analysis - Crimes by Minute

# In[8]:

minute = train['Minute'].unique()
data_month_minute = pd.crosstab(train['Minute'], train['Month'])
axhandles=data_month_minute.plot(kind='bar',subplots=True,layout=(4,3),figsize=(16,12),sharex=True,sharey=True,xticks=range(0,60,4),rot=0)
i=1
for axrow in axhandles:
    for ax in axrow:
        ax.set_xticklabels(range(0,60,1))
        ax.legend([month_dict[i]],loc='best')
        ax.set_title("")
        i+=1
plt.suptitle('Distribution of Crimes by Minutes',size=20)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('Distribution_of_Crimes_by_Minutes.png')


# ![title](Distribution_of_Crimes_by_Minutes.png)

# ## Data Analysis - Day of Year

# In[12]:

train = pd.read_csv("../data/train.csv", parse_dates=['Dates'])
train = train.drop(['Descript', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'], axis=1)
train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))
train.head()


# In[12]:

#Add day of the year format 02-22
df = train[['Category','DayOfYear']].groupby(['DayOfYear']).count()

df.plot(y='Category', label='Number of events', figsize=(10,5)) 
plt.title("Crimes occur with a regular pattern: two peaks per month")
plt.ylabel('Number of crimes')
plt.xlabel('Day of year')
plt.grid(True)

plt.savefig('Distribution_of_Crimes_by_DayofYear.png')


# ![title](Distribution_of_Crimes_by_DayofYear.png)

# ## Prediction

# In[111]:

train.head()


# In[112]:

train.loc[train["Year"]==2015,"Year"] = 2014
train.loc[train["Minute"]!=(0|30),"Minute"] = 0
train.head()


# In[113]:

from sklearn.cross_validation import cross_val_score

feature_names = ["Year", "Month", "Minute", "Days", "Hour"]
label_name = "Category"

train_X = train[feature_names]
train_y = train[label_name]


# ### Scoring

# In[114]:

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gaussian_score = cross_val_score(GaussianNB(), train_X, train_y, scoring='log_loss', cv=5).mean()
bernoulli_score = cross_val_score(BernoulliNB(), train_X, train_y, scoring='log_loss', cv=5).mean()

print("GaussianNB = {0:.6f}".format(-1.0 * gaussian_score))
# print("MultinomialNB = {0:.6f}".format(-1.0 * multimonial_score))
# print("BernoulliNB = {0:.6f}".format(-1.0 * bernoulli_score))


# ## Test Results

# ### Before
#  - Year & Minute : 2.704256
#  - Hour & Minute : 2.633491
#  - Month & Days : 2.682305
#  - Days & Hour : 2.666096
#  - Year & Month & Days & Hour & Minute : 2.691690
#  
# ### After
#  - Year & Minute : 2.973289 (+0.269033)
#  - Hour & Minute : 2.879516 (+0.246025)
#  - Year & Month & Days & Hour & Minute : 2.940873 (+0.249183)
