#!/usr/bin/env python
# coding: utf-8

# ## Decision Tree with ID3 Algorithm

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\data-set for decision tree.csv")
df.head()


# In[4]:


df.describe()


# ### Numerical and Categorical Columns

# In[5]:


categorical_features = df.select_dtypes(include=['object'])
categorical_features.head(1)


# In[6]:


categorical_features = list(categorical_features)
print(f'Categorical Features are : {categorical_features}')


# In[7]:


# There are no numerical features in our dataset.
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
print(f'numerical features are : {numerical_features}')


# ### NAN Values

# In[8]:


#There are no null values there is "?" that might affect the data
df.isnull().sum()  # Count missing values per column


# In[10]:


# In some rows  there are alot of "?" that can significantly affect our output so for that we'lll use forward fill
(df == '?').sum()


# In[11]:


#df.replace('?', np.nan, inplace=True)  # Convert '?' to NaN


# ###  Encoding

# In[12]:


df = pd.get_dummies(df, columns=df.columns.difference(['income']), drop_first=True)


# ### Training and Testing

# In[13]:


#  Split Data into Features (X) and Target (y) 

X = df.drop(columns=['income'])
y = df[['income']]


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[15]:


model = DecisionTreeClassifier(criterion='entropy')  


# In[16]:


model.fit(X_train,y_train)


# In[17]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[18]:


accuracy


# In[20]:


# Plot accuracy for different depths
depths = range(1, 21)
scores = [DecisionTreeClassifier(criterion='entropy', max_depth=d).fit(X_train, y_train).score(X_test, y_test) for d in depths]

plt.plot(depths, scores, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth  and Accuracy')
plt.show()


# In[ ]:




