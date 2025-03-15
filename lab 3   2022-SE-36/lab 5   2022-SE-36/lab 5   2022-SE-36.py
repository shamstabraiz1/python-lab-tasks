#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with One Variable:

# Dataset of 1000 records..

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[7]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\python lab\\taxi_trip_pricing   (Dataset for Lab 5 (Linear Regression With One Variable).csv")
df.head()


# In[8]:


df.describe()


# In[10]:


sns.scatterplot(x='Trip_Distance_km',y='Trip_Price',data=df)
# The Scatter plot shows that increase in Trip_Distance_km, Trip_Price also increases.
# linear regression with one variable on Trip_Price and Trip_Distance_km


# NAN Values Handling 

# In[11]:


# Handling NAN values present in distance.

df.shape


# In[12]:


df['Trip_Distance_km'].count()

# Trip_Distance_km is 950 it means that we have NAN values in this feature. 


# In[13]:


df['Trip_Distance_km'].mean()


# In[14]:


df['Trip_Distance_km'].median()


# In[15]:


df['Trip_Distance_km'].mode()

# So for NAN values We will use median to fill NAN values of T_D_km


# In[16]:


Q1 = df['Trip_Distance_km'].quantile(0.25)  # 1st quartile
Q3 = df['Trip_Distance_km'].quantile(0.75)  # 3rd quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[18]:


# Impute missing values with the median
df['Trip_Distance_km'].fillna(df['Trip_Distance_km'].median(), inplace=True)


# Handling Outliers

# In[27]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Trip_Distance_km'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# In[28]:


Q1 = df['Trip_Distance_km'].quantile(0.25)
Q3 = df['Trip_Distance_km'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f'Lower Bond : {lower_bound} \nUpper Bond : {upper_bound}')


# In[29]:


df[df['Trip_Distance_km'] <= upper_bound]['Trip_Distance_km'].max()


# In[31]:


df['Trip_Distance_km'] = df['Trip_Distance_km'].apply(lambda x: 25.83 if x > upper_bound else x)


# In[32]:


sns.boxplot(data=df['Trip_Distance_km'], color='lightblue')
plt.title('Boxplot')
plt.show()

# new box plot with no outlier.


# ## Taxi price

# NAN Values Handling

# In[35]:


df['Trip_Price'].count()
#it means that we have again NAN values in the feature.


# In[36]:



df['Trip_Price'].mean()


# In[37]:



df['Trip_Price'].median()


# In[38]:


df['Trip_Price'].mode()


# In[39]:


Q1 = df['Trip_Price'].quantile(0.25)  # First quartile
Q3 = df['Trip_Price'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[40]:


df['Trip_Price'].fillna(df['Trip_Price'].median(), inplace=True)


# ### Handling Outliers:

# In[41]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Trip_Price'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# In[42]:



Q1 = df['Trip_Price'].quantile(0.25)
Q3 = df['Trip_Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f'Lower Bond : {lower_bound} \nUpper Bond : {upper_bound}')


# In[43]:



df[df['Trip_Price'] <= upper_bound]['Trip_Price'].max()


# In[44]:


df['Trip_Price'] = df['Trip_Price'].apply(lambda x: 50 if x > upper_bound else x)


# In[46]:


sns.boxplot(data=df['Trip_Price'], color='lightblue')
plt.title('Boxplot')
plt.show()

# Trip_Price with no outliers


# # Training and Testing

# In[47]:


scaler = StandardScaler()
df['Trip_Distance_km'] = scaler.fit_transform(df[['Trip_Distance_km']])
df['Trip_Price'] = scaler.fit_transform(df[['Trip_Price']])


# In[48]:


X = df[['Trip_Distance_km']]
y = df['Trip_Price']


# In[49]:



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[50]:


model = LinearRegression()


# In[51]:


model.fit(X_train,y_train)


# In[52]:


y_pred = model.predict(X_test)


# In[53]:


mse = mean_squared_error(y_test,y_pred)


# In[54]:


mse


# In[55]:


np.sqrt(mse)


# In[57]:


np.sqrt(mse)


# In[58]:


# Create a scatter plot to compare actual vs predicted prices
plt.figure(figsize=(8, 5))  # Set the figure size
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Actual Prices')  # Scatter plot of actual vs predicted

# Add a perfect prediction line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Prediction')

# Add labels, title, and legend
plt.xlabel('Actual Prices')  # X-axis label
plt.ylabel('Predicted Prices')  # Y-axis label
plt.title('Comparison of Actual and Predicted Prices')  # Plot title
plt.legend()  # Show legend
plt.grid(True)  # Enable grid for better readability
plt.show()  # Display the plot


# In[59]:


print(f"\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")


# In[ ]:




