#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


housing=pd.read_csv("Book2.csv")


# In[4]:


housing.head()


# In[5]:


# housing.info()


# In[6]:


# housing['CHAS'].value_counts()


# In[7]:


# housing.describe()


# In[8]:


# import matplotlib.pyplot as plt


# In[9]:


# housing.hist(bins=50,figsize=(20,15))


# ## Train Test splitting

# In[10]:


# # For learning purpose
# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:] 
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


# from sklearn.model_selection import train_test_split
# train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[12]:


# from sklearn.model_selection import StratifiedShuffleSplit
# split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
# for train_indices,test_indices in split.split(housing,housing['CHAS']):
#     strat_train_set=housing.loc[train_indices]
#     strat_test_set=housing.loc[test_indices]    


# In[13]:


# print(f"Rows in train set:",len(train_set),"\n","Rows test set:",len(test_set))


# # Looking for correlations

# In[14]:


# corr_matrix=housing.corr()
# corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


# from pandas.plotting import scatter_matrix
# attributes=["MEDV","RM","ZN","LSTAT"]
# scatter_matrix(housing[attributes],figsize=(12,8))


# In[16]:


# housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.5)


# In[17]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_indices,test_indices in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_indices]
    strat_test_set=housing.loc[test_indices]
 



# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
myPipeline=Pipeline([
    
    ('std_scaler',StandardScaler())
])


# # Creating a pipeline

# In[19]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"]


# In[20]:


housing_mean_transform=myPipeline.fit_transform(housing)


# # Selecting a desired model for prediction

# In[21]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(housing_mean_transform,housing_labels)


# # Cross Validation

# In[22]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_mean_transform,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
print(rmse_scores)


# In[23]:


def print_scores(scores):
    print(scores.mean())
    print(scores.std())
    


# In[24]:


print_scores(rmse_scores)


# In[25]:


from joblib import dump,load
dump(model,"Dragon.joblib")


# # Testing the model

# In[26]:


x_unprepared=strat_test_set.drop("MEDV",axis=1)
y_real=strat_test_set["MEDV"].copy()
x=myPipeline.fit_transform(x_unprepared)


# In[27]:


y_predicted=model.predict(x)


# In[28]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_real,y_predicted)
rmse=np.sqrt(mse)


# In[29]:


rmse


# In[ ]:




