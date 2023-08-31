#!/usr/bin/env python
# coding: utf-8

# ### Creating a logistic regression to predict absenteeism

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')


# In[3]:


data_preprocessed.drop(['Date'], axis=1)


# In[4]:


data_preprocessed.head(5)


# In[5]:


data_preprocessed.head(5)


# In[6]:


data_preprocessed['Absenteeism Time in Hours'].median()


# In[7]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)


# In[8]:


targets


# In[9]:


data_preprocessed['Excessive Absenteeism'] = targets


# In[10]:


data_preprocessed.head()


# In[11]:


targets.sum()


# In[12]:


targets.shape[0]


# In[13]:


targets.sum() / targets.shape[0]


# In[14]:


data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the Week', 'Daily Work Load Average', 
                                            'Distance to Work'], axis=1)


# In[15]:


data_with_targets = data_with_targets.drop(['Date'], axis=1)


# In[16]:


data_with_targets is data_preprocessed


# In[17]:


data_with_targets.head()


# ### Select the imnputs for the regression

# In[18]:


data_with_targets.shape


# In[19]:


data_with_targets.iloc[:, 0:14 ]


# In[20]:


data_with_targets.iloc[:, :-1 ]


# In[21]:


unscaled_inputs = data_with_targets.iloc[:, :-1 ]


# In[22]:


unscaled_inputs


# ### Standardize the data

# In[23]:


# from sklearn.preprocessing import StandardScaler
# absenteeism_scaler = StandardScaler()


# In[24]:


# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
# moreover, one of them is actually the StandardScaler module, 
# so you can imagine that the Custom Scaler is build on it

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler()
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[25]:


unscaled_inputs.columns.values


# In[26]:


# columns_to_scale = ['Transportation Expense', 'Distance to Work', 'Age',
#        'Daily Work Load Average', 'Body Mass Index','Children', 'Pets', 'Month Value', 'Day of the Week']
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']


# In[27]:


columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[28]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


absenteeism_scaler.fit(unscaled_inputs)


# In[30]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[31]:


# for transforming new data when deploying a model
# new_data_raw = pd.read_csv('new_data.csv')
# new_data_scaled = absenteeism_scaler.transform(new_data_raw)


# In[32]:


scaled_inputs


# In[33]:


scaled_inputs.shape


# ## Split the data into train & test and shuffle

# In[34]:


from sklearn.model_selection import train_test_split


# ### Split

# In[35]:


train_test_split(scaled_inputs, targets)


# We get 4 arrays:
# 1. a training dataset with inputs
# 2. a training dataset with targets
# 3. a test dataset with inputs
# 4. a test dataset with targets

# In[36]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, shuffle=True, random_state=20)


# In[37]:


print(x_train.shape, y_train.shape)


# In[38]:


print(x_test.shape, y_test.shape)


# ### Logistic regression with sklearn

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training the model

# In[40]:


reg = LogisticRegression()


# In[41]:


reg.fit(x_train, y_train)


# In[42]:


print(reg)


# In[43]:


reg.score(x_train, y_train) # receive inputs and targets


# ### Manually check the accuracy

# sklearn.linear_model.LogisticRegression.predict(inputs)
# predicts class label (logistic regression outputs) for given input samples

# In[44]:


model_outputs = reg.predict(x_train)


# In[45]:


model_outputs


# In[46]:


y_train #targets


# In[47]:


model_outputs == y_train


# In[48]:


np.sum(model_outputs == y_train)


# In[49]:


model_outputs.shape[0]


# So, this is the same result that reg.score(x_train, y_train)

# In[50]:


np.sum(model_outputs == y_train) / model_outputs.shape[0]


# ### Finding xthe intercept and coefficients

# In[51]:


reg.intercept_


# In[52]:


reg.coef_


# 
# Whenever we employ sklearn (usually) the results are arrays, not data frames
# That's why this throws an error: 
# scaled_inputs.columns.values

# In[53]:


unscaled_inputs.columns.values


# In[54]:


feature_name = unscaled_inputs.columns.values


# In[55]:


summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table


# In this way we will 'shift' up all indices by 1

# In[56]:


summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# ### Interpreting the coefficients

# bias / intercept     coefficient / weight
# The closer they [the weights] to 0, the smaññer the weight
# Whochever weights is bigger, its corresponding feature is more important
# 
# Standardized coefficients are basically:
# the coefficients of a regression where all variables have been standardized

# In[57]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[58]:


summary_table


# In[59]:


summary_table.sort_values('Odds_ratio', ascending=False)


# 

# ### Testing the model

# In[60]:


reg.score(x_test, y_test)


# In[61]:


predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[62]:


predicted_proba.shape


# In[63]:


predicted_proba[:, 1]


# ### Save the model

# In[64]:


import pickle


# In[66]:


with open('model', 'wb') as file:
    pickle.dump(reg, file)


# In[67]:


# pickle the scaler file
with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)


# In[ ]:





# In[ ]:




