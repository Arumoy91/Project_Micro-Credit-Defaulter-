#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Datafile_telecom.csv')
df


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


dfcorr=df.corr()
dfcorr


# In[11]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[12]:


df.drop('pcircle',axis=1,inplace=True)


# In[13]:


df.drop('pdate',axis=1,inplace=True)


# In[14]:


df.head()


# In[15]:


sns.countplot(df['label'])


# In[16]:


plt.figure(figsize=(32,16))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0)


# In[17]:


sns.relplot(x='aon',y="maxamnt_loans30",hue='label',data=df)


# In[18]:


sns.relplot(x='payback30',y="maxamnt_loans30",hue='label',data=df)


# In[19]:


sns.relplot(x='payback90',y="maxamnt_loans90",hue='label',data=df)


# In[20]:


from sklearn.preprocessing import LabelEncoder

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# In[21]:


df = df.apply(lambda x: object_to_int(x))
df.head()


# In[22]:


X=df.drop(['label'], axis=1)
y=df['label']


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 42)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


rf = RandomForestClassifier()


# In[27]:


rf.fit(X_train,y_train)


# In[28]:


rf.score(X_test,y_test)


# In[29]:


df = DecisionTreeClassifier()


# In[30]:


df.fit(X_train,y_train)


# In[31]:


df.score(X_test,y_test)


# In[32]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[33]:


gb = GradientBoostingClassifier()


# In[34]:


gb.fit(X_train,y_train)


# In[35]:


gb.score(X_test,y_test)


# In[36]:


ad=AdaBoostClassifier()


# In[37]:


ad.fit(X_train,y_train)


# In[38]:


ad.score(X_test,y_test)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


lr= LogisticRegression()


# In[41]:


lr.fit(X_train,y_train)


# In[42]:


lr.score(X_test,y_test)


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[45]:


y_pred = lr.predict(X_test)


# In[46]:


from sklearn.metrics import confusion_matrix,classification_report


# In[47]:


print(classification_report(y_test, y_pred))


# In[48]:


cm=confusion_matrix(y_test,ad.predict(X_test))


# In[49]:


print(cm)


# In[50]:


from sklearn.externals import joblib 


# In[51]:


joblib.dump(rf, 'datafile.pkl') 


# In[52]:


rf_from_joblib = joblib.load('datafile.pkl')  


# In[53]:


rf_from_joblib.predict(X_test) 


# In[ ]:




