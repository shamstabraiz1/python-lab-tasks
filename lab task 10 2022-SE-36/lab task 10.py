#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer, KNNImputer


# In[3]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\python lab\\advertising_ef.csv")
df.head()


# In[4]:



df.info()


# In[5]:



df.describe()


# ## NAN values

# In[6]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# In[7]:



df.isnull().sum()


# In[8]:



# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


# In[9]:


# Handle missing values
num_imputer = KNNImputer(n_neighbors=3)
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


# In[10]:


df.isnull().sum()


# ### Categorical Columns

# In[11]:


cat_cols


# In[13]:


label_encoder = LabelEncoder()

for features in cat_cols:
    df[features] = label_encoder.fit_transform(df[features])


# ### Outliers in Numerical  Column

# In[14]:



num_cols


# In[16]:


plt.figure(figsize=(10,8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot for Outlier')
plt.show()


# In[17]:


# figure size
plt.figure(figsize=(15, 8))

# Creating boxplots
for i, feature in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)  # Creating a 2-row, 4-column grid of subplots
    sns.boxplot(y=df[feature], color='skyblue')  # Boxplot for each feature
    plt.title(f'Boxplot for {feature}')  # Title for each boxplot

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[18]:


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1                   # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR    # Lower whisker
    upper_bound = Q3 + 1.5 * IQR    # Upper whisker

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers_columns= []

for col in num_cols:
    outliers = detect_outliers_iqr(df, col)
    if outliers.empty:
        print(f'Column {col} has no outliers')
    else:
        print(f"Column: {col}")
        print(f"Number of Outliers: {len(outliers)}")
        outliers_columns.append(col)
    print('\n')
    


# ### Scaling Numerical Column

# In[19]:


df[num_cols].hist(figsize=(10, 6), bins=30)
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[20]:


# Standardize numerical features
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# In[21]:



df[num_cols].hist(figsize=(10, 6), bins=30)
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# ## Training and Testing

# In[22]:


X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']


# In[23]:


num_cols.remove('Clicked on Ad')


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[25]:


# Train GaussianNB for numerical columns
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train[num_cols], y_train)


# In[26]:



# Train CategoricalNB for categorical columns
categorical_nb = MultinomialNB()
categorical_nb.fit(X_train[cat_cols], y_train)


# In[27]:



# Predict probabilities
num_probs = gaussian_nb.predict_proba(X_test[num_cols])
cat_probs = categorical_nb.predict_proba(X_test[cat_cols])


# In[28]:



# Combine predictions by multiplying probabilities
final_probs = num_probs * cat_probs
final_preds = final_probs.argmax(axis=1)


# In[29]:



# Evaluate
accuracy = accuracy_score(y_test, final_preds)
print(f'Combined Model Accuracy: {accuracy:.2f}')


# In[30]:


print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))


# In[31]:


print("Classification Report:\n", classification_report(y_test, final_preds))


# In[34]:


# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, final_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Ensemble Confusion Matrix")
plt.show()


# In[ ]:




