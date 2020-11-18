#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns


# In[2]:


df= pd.read_csv("AutoData.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.describe()


# In[7]:


df.describe(include="object")


# ### Univariate Analysis

# In[8]:


df.price.plot.box()
plt.show()


# In[14]:


df.make.value_counts(normalize= True)


# In[15]:


df.fueltype.value_counts(normalize= True)


# In[16]:


df.aspiration.value_counts(normalize= True)


# In[17]:


df.doornumber.value_counts(normalize= True)


# In[18]:


df.carbody.value_counts(normalize= True)


# In[19]:


df.drivewheel.value_counts(normalize= True)


# In[20]:


df.enginelocation.value_counts(normalize= True)


# In[21]:


df.enginetype.value_counts(normalize= True)


# In[22]:


df.cylindernumber.value_counts(normalize= True)


# In[23]:


df.fuelsystem.value_counts(normalize= True)


# ### Mulivariate Analysis

# In[26]:


df.groupby(['fueltype'])["price"].mean()


# In[27]:


df.groupby(['aspiration'])["price"].mean()


# In[28]:


df.groupby(['carbody'])["price"].mean()


# In[29]:


df.groupby(['doornumber'])["price"].mean()


# In[30]:


df.groupby(['drivewheel'])["price"].mean()


# In[31]:


df.groupby(['enginelocation'])["price"].mean()


# In[32]:


df.groupby(['cylindernumber'])["price"].mean()


# In[33]:


df.groupby(['fuelsystem'])["price"].mean()


# In[34]:


sns.pairplot(df, x_vars=['curbweight','enginesize','boreratio','stroke','compressionratio'],y_vars='price', markers="+", diag_kind="scatter")
plt.show()


# In[35]:


sns.pairplot(df, x_vars=['symboling','wheelbase','carlength','carwidth','carheight'],y_vars='price', markers="+", diag_kind="scatter")
plt.show()


# In[37]:


sns.pairplot(df, x_vars=['horsepower','peakrpm','citympg','highwaympg'],y_vars='price', markers="+", diag_kind="scatter")
plt.show()


# ### Encoding Categorical Varaibles

# In[38]:


df1= df.copy()


# In[39]:


from sklearn.preprocessing import LabelEncoder


# In[43]:


le= LabelEncoder()
df1["symboling"]= le.fit_transform(df1["symboling"])
df1.symboling.unique()


# In[44]:


df1["make"]= le.fit_transform(df1["make"])
df1.make.unique()


# In[45]:


df1["fueltype"]= le.fit_transform(df1["fueltype"])
df1.fueltype.unique()


# In[46]:


df1["aspiration"]= le.fit_transform(df1["aspiration"])
df1.aspiration.unique()


# In[47]:


df1["doornumber"]= le.fit_transform(df1["doornumber"])
df1.doornumber.unique()


# In[48]:


df1["carbody"]= le.fit_transform(df1["carbody"])
df1.carbody.unique()


# In[49]:


df1["drivewheel"]= le.fit_transform(df1["drivewheel"])
df1.drivewheel.unique()


# In[50]:


df1["enginelocation"]= le.fit_transform(df1["enginelocation"])
df1.enginelocation.unique()


# In[51]:


df1["enginetype"]= le.fit_transform(df1["enginetype"])
df1.enginetype.unique()


# In[52]:


df1["fuelsystem"]= le.fit_transform(df1["fuelsystem"])
df1.fuelsystem.unique()


# In[53]:


df1["cylindernumber"]= le.fit_transform(df1["cylindernumber"])
df1.cylindernumber.unique()


# In[54]:


df1.head()


# ### Corelation with HeatMap

# In[58]:


plt.figure(figsize=(16,16))
sns.heatmap(df1.corr(), annot= True, fmt= ".0%")
plt.show()


# In[59]:


plt.figure(figsize=(16,16))
sns.heatmap(df.corr(), annot= True, fmt= ".0%")
plt.show()


# In[65]:


sns.jointplot('horsepower', 'price', data = df, kind="reg")
plt.show()


# In[64]:


sns.jointplot('enginesize', 'price', data = df, kind="reg")
plt.show()


# In[67]:


sns.jointplot('curbweight', 'price', data = df, kind="reg")
plt.show()


# ### From Heat map, pair plot and joint plot, we can conclude that curbweight, enginesize and horsepower are most correlated with price. So let's go ahead and perform simple linear regression using enginesize as our feature variable. Since its seems most correlated with price.

# ### Simple Linear Regression

# In[68]:


X= df1[["enginesize"]]
y= df1[["price"]]


# ### Train_Test_Split

# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[72]:


X_train.shape, X_test.shape


# ### Model Building

# In[73]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train,y_train)


# In[74]:


model.score(X_train,y_train)


# In[75]:


model.intercept_, model.coef_


# In[76]:


model.score(X_test,y_test)


# ### From the parameters that we get, our linear regression equation becomes:
# 
# ### price=-8442.446+171.001×enginesize

# ### Multiple Linear Regression

# In[79]:


X= df1.drop("price", axis=1)
y= df1["price"]


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[81]:


X_train.shape, X_test.shape


# ### Model Building

# In[82]:


from sklearn.linear_model import LinearRegression
mod= LinearRegression()
mod.fit(X_train,y_train)


# In[83]:


mod.intercept_, mod.coef_


# In[85]:


mod.score(X_train,y_train)


# In[86]:


mod.score(X_test,y_test)


# In[ ]:




