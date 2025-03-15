#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression..

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelEncoder


# In[5]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\loan_data.csv")
df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

# There are no NAN values in the Dataset


# ## Label Encod

# In[9]:


categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']


# In[10]:


label_encoder = LabelEncoder()

for col in categorical_features:
    df[col] = label_encoder.fit_transform(df[col])


# In[11]:


for col in categorical_features:
    print(df[col].unique())


# ### Outliers

# In[13]:


for graph in df:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df[graph], color='lightblue')
    plt.title(graph)
    plt.xticks(rotation=45)
    plt.show()
    
# plt.figure(figsize=(10,8))
# sns.boxplot(data=df, color='lightblue')
# plt.title('Boxplot')
# plt.xticks(rotation=45)


# In[14]:


outlier_features =  ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']


# In[17]:


for feature in outlier_features:
    Q1 = df[feature].quantile(0.25)  # First quartile
    Q3 = df[feature].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    median = df[feature].median()
    
    # Replace values outside the whiskers with the median
    df[feature] = df[feature].apply(lambda x: median if x > upper_whisker or x < lower_whisker else x)


# In[18]:


outliers = df[(df[feature] < lower_whisker) | (df[feature] > upper_whisker)]
print(f"{feature}: {len(outliers)} outliers")


# In[19]:


for graph in df:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df[graph], color='lightblue')
    plt.title(graph)
    plt.xticks(rotation=45)
    plt.show()


# ## Scaling Data

# In[20]:


#  Split Data into Features (X) and Target (y) 

X = df.drop(columns=['loan_status'])
y = df[['loan_status']]


# In[21]:


#  Scaling Data

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[22]:


# Split Data for Training and Testing

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[23]:


# Model Selection

model = LogisticRegression()


# In[24]:


# Train Logistic Regression Model

model.fit(X_train,y_train)


# In[26]:


# Check Model Coefficients

model.coef_


# In[27]:


# Check Model Intercept

model.intercept_


# In[28]:


# Evaluate Model Performance

model.score(X_test, y_test)


# In[29]:


y_test


# In[32]:


# Predictions

y_predict = model.predict(X_test)


# In[33]:



y_predict


# In[36]:


probabilities = model.predict_proba(X_test)
probabilities


# In[37]:


# Evaluate Predictions

report  = classification_report(y_test, y_predict)


# In[38]:



print(report)


# In[39]:


confusion_matrix(y_test, y_predict)


# In[43]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## Accuracy

# In[44]:



Accuracy = (6506+1446)/(6506+1446+505+543)
print(f'Accuracy : {Accuracy}')


# ## Precision

# In[45]:


# Class 0
precision_0 = 6506/(6506+505)
print(f'Precision of Class 0 : {precision_0}')


# In[46]:



# Class 1
precision_1 = 1446/(1446+505)
print(f'Precision of Class 0 : {precision_1}')


# ## Recall

# In[47]:


# Class 0
Recall_0 = 6506/(6506+543)
print(f'Recall of Class 0 : {Recall_0}')


# In[48]:


# Class 1
Recall_1 = 1446/(1446+543)
print(f'Recall of Class 0 : {Recall_1}')


# In[ ]:




