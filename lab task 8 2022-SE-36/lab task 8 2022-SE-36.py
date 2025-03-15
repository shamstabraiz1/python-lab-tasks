#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\lab task 8 2022-SE-36\\heart.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Handling NAN values

# In[6]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

# There is no null values in this dataset


# ## Label Encod

# In[7]:


categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']


# In[8]:


label_encoder = LabelEncoder()

for features in categorical_features:
    df[features] = label_encoder.fit_transform(df[features])


# In[9]:


for features in categorical_features:
    print(df[features].unique())


# In[10]:


df.describe()


# ###  Outliers

# In[12]:


plt.figure(figsize=(10,8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot for Outliers ')
plt.show()


# In[13]:


numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease' ]


# In[33]:


# Setting the figure size
plt.figure(figsize=(15, 8))

# Creating boxplots
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)  # Creating a 2-row, 4-column grid of subplots
    sns.boxplot(y=df[feature], color='skyblue')  # Boxplot for each feature
    plt.title(f'Boxplot for {feature}')  # Title for each boxplot

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[36]:


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1                   # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR    # Lower whisker
    upper_bound = Q3 + 1.5 * IQR    # Upper whisker

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for col in numerical_features:
    outliers = detect_outliers_iqr(df, col)
    if outliers.empty:
        print(f'Column {col} has no outliers')
    else:
        print(f"Column: {col}")
        print(f"Number of Outliers: {len(outliers)}")
        outliers_columns.append(col)
    print('\n')


# In[47]:


numerical_features


# In[48]:


for col in numerical_features:
    print(f'Column : {col}')
    print(f'mean : {df[col].mean()}')
    print(f'median : {df[col].median()}')
    print(f'mode : {df[col].mode()}')
    Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1                   # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR    # Lower whisker
    upper_bound = Q3 + 1.5 * IQR    # Upper whisker
    print(f'IQR : {IQR}')
    print(f'lower bond : {lower_bound}')
    print(f'upper bond : {upper_bound}')
    print('\n')


# In[49]:


numerical_df = df[numerical_features]


# In[50]:


numerical_df.describe()


# In[51]:


def impute_outliers_with_mean(df, column):
    Q1 = df[column].quantile(0.25)  
    Q3 = df[column].quantile(0.75)  
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR  
    upper_bound = Q3 + 1.5 * IQR  

    mean_value = df[column].mean()
    
    df[column] = df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
    return df

# Apply to numerical columns with outliers
for col in numerical_features:
    df = impute_outliers_with_mean(df, col)

print("Outliers replaced with mean successfully!")


# In[52]:


#figure size
plt.figure(figsize=(15, 8))

# Creating boxplots
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 4, i)  # Creating a 2-row, 4-column grid of subplots
    sns.boxplot(y=df[feature], color='skyblue')  # Boxplot for each feature
    plt.title(f'Boxplot for {feature}')  # Title for each boxplot

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[53]:


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1                   # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR    # Lower whisker
    upper_bound = Q3 + 1.5 * IQR    # Upper whisker

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for col in numerical_features:
    outliers = detect_outliers_iqr(df, col)
    if outliers.empty:
        print(f'Column {col} has no outliers')
    else:
        print(f"Column: {col}")
        print(f"Number of Outliers: {len(outliers)}")
        outliers_columns.append(col)
    print('\n')


# ## Scaling Data

# In[54]:


df.hist(figsize=(10, 6), bins=30)
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[55]:


#  Split Data into Features (X) and Target (y) 

X = df.drop(columns=['HeartDisease'])
y = df[['HeartDisease']]


# In[56]:


scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# ## Training

# In[57]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[58]:


knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)


# In[59]:


knn.predict(X_test)


# In[60]:


accuracy_score(y_test,knn.predict(X_test))


# In[61]:


k_values = range(1, 50,2)


# In[63]:


accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot K vs Accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='dashed', color='g')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Choosing the Best K")
plt.xticks(k_values)
plt.grid(True)
plt.show()
plt.tight_layout()
# Find best K
best_k = k_values[np.argmax(accuracy_scores)]
print(f"Best K: {best_k}")


# In[ ]:




