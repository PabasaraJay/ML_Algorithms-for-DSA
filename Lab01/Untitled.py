#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[4]:


zoo_data=pd.read_csv("zoo.csv")
zoo_data.head()


# In[5]:


zoo_data.shape


# In[9]:


x=zoo_data.iloc[:,1:17]


# In[10]:


x.shape


# In[11]:


x.head()


# In[13]:


y=zoo_data.iloc[:,17]


# In[14]:


y.shape


# In[15]:


y.head()


# In[16]:


X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.25)


# In[20]:


X_train.shape


# In[21]:


X_train.head()


# In[22]:


X_test.shape


# In[23]:


X_test.head()


# In[24]:


y_train.shape


# In[25]:


y_train.head()


# In[26]:


y_test.shape


# In[28]:


y_test.head()


# In[29]:


zoo_classifier = DecisionTreeClassifier(random_state=0)


# In[30]:


zoo_classifier.fit(X_train,y_train)


# In[38]:


zoo_classifier.score(X_test,y_test)


# In[40]:


zoo_classifier.predict(X_test[10:15])


# In[ ]:




