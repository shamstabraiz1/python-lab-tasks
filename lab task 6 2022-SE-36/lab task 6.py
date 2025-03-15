#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with multiple variables

# In[218]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[219]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# In[220]:


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\taxi_trip_pricing   (Dataset for Lab 5 (Linear Regression With One Variable).csv")
df.head()


# In[221]:


df.describe()


# In[222]:


df.shape


# In[223]:


df.columns


# In[224]:


# Hanling NAN Values:

# with the median
df['Trip_Distance_km'].fillna(df['Trip_Distance_km'].median(), inplace=True)


# In[225]:



# Handling Outliers:

df['Trip_Distance_km'] = df['Trip_Distance_km'].apply(lambda x: 25.83 if x > 74 else x)


# ## Time

# In[226]:


df['Time_of_Day'].count()

#Heres 50 missing values in Time_of_Day


# In[227]:



df['Time_of_Day'].unique()

#present  values  in Time_of_Day


# ## Handling Misssing Values

# In[228]:


df['Time_of_Day'].mode()


# In[229]:



df['Time_of_Day'].fillna('Afternoon', inplace=True)
df['Time_of_Day'].count()


# In[230]:


label = LabelEncoder()
df['Time_of_Day'] = label.fit_transform(df['Time_of_Day'])
df['Time_of_Day'].unique()


# ## Day_of_Week

# In[231]:


df['Day_of_Week'].count()

# Heres 50 missing values in Day_of_Week


# In[232]:


df['Day_of_Week'].unique()


# In[233]:


df['Day_of_Week'].mode()


# In[234]:


df['Day_of_Week'].fillna('Weekday', inplace=True)
df['Day_of_Week'].count()


# In[235]:


df['Day_of_Week'] = label.fit_transform(df['Day_of_Week'])
df['Day_of_Week'].unique()


# ## Passengers_Count

# NAN values

# In[236]:


df['Passenger_Count'].mean()


# In[237]:



df['Passenger_Count'].median()


# In[238]:


df['Passenger_Count'].mode()


# In[239]:


Q1 = df['Passenger_Count'].quantile(0.25)  # First quartile
Q3 = df['Passenger_Count'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[240]:



df['Passenger_Count'].fillna(df['Passenger_Count'].mean(), inplace=True)


# In[241]:


df['Passenger_Count'].count()


# ### Handling Outliers:

# In[242]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Passenger_Count'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# ## Traffic_Conditions

# In[243]:


df['Traffic_Conditions'].count()

# Heres 50 missing values in Traffic_Conditions


# In[244]:


df['Traffic_Conditions'].unique()

#present values in Time_of_Day


# ### Handling missing values

# In[245]:


df['Traffic_Conditions'].mode()


# In[246]:


df['Traffic_Conditions'].fillna('Low', inplace=True)
df['Traffic_Conditions'].count()


# In[247]:


df['Traffic_Conditions'] = label.fit_transform(df['Traffic_Conditions'])


# In[248]:


df['Traffic_Conditions'].unique()


# ## Weather

# In[249]:


df['Weather'].count()

#Heres 50 missing values in Time_of_Day


# In[250]:


df['Weather'].unique()


# ### Handling missing vales

# In[251]:



df['Weather'].mode()


# In[252]:


df['Weather'].fillna('Clear', inplace=True)


# In[253]:



df['Weather'].count()


# In[254]:


df['Weather'] = label.fit_transform(df['Weather'])


# In[287]:


df['Weather'].unique()


# ## Base_Fare

# In[288]:


df['Base_Fare'].count()


# ###  missing values

# In[289]:


df['Base_Fare'].mean()


# In[290]:



df['Base_Fare'].median()


# In[291]:



df['Base_Fare'].mode()


# In[292]:


Q1 = df['Base_Fare'].quantile(0.25)  # First quartile
Q3 = df['Base_Fare'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[293]:


df['Base_Fare'].fillna(df['Base_Fare'].mean(), inplace=True)


# In[294]:


df['Base_Fare'].count()


# In[296]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Base_Fare'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# In[ ]:





# ## Rate Per Km 

# In[256]:


df['Per_Km_Rate'].count()

#Heres 50 missing values in Time_of_Day


# ### missing values

# In[257]:


df['Per_Km_Rate'].mean()


# In[258]:


df['Per_Km_Rate'].median()


# In[259]:


df['Per_Km_Rate'].mode()


# In[260]:


Q1 = df['Per_Km_Rate'].quantile(0.25)  # First quartile
Q3 = df['Per_Km_Rate'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[261]:



df['Per_Km_Rate'].fillna(df['Per_Km_Rate'].mean(), inplace=True)


# In[262]:



df['Per_Km_Rate'].count()


# ## Outliers

# In[263]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Per_Km_Rate'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# ##  Rate Per Minute

# In[264]:


df['Per_Minute_Rate'].count()


# ###  missing values

# In[265]:



df['Per_Minute_Rate'].mean()


# In[266]:


df['Per_Minute_Rate'].median()


# In[267]:


df['Per_Minute_Rate'].mode()


# In[268]:


Q1 = df['Per_Minute_Rate'].quantile(0.25)  # First quartile
Q3 = df['Per_Minute_Rate'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[269]:


df['Per_Minute_Rate'].fillna(df['Per_Minute_Rate'].mean(), inplace=True)


# In[270]:


df['Per_Minute_Rate'].count()


# ### Outliers

# In[271]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Per_Minute_Rate'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# ## Trip Duration Minutes

# In[272]:



df['Trip_Duration_Minutes'].count()


# ### missing values

# In[273]:


df['Trip_Duration_Minutes'].mean()


# In[274]:


df['Trip_Duration_Minutes'].median()


# In[275]:


df['Trip_Duration_Minutes'].mode()


# In[276]:


Q1 = df['Trip_Duration_Minutes'].quantile(0.25)  # First quartile
Q3 = df['Trip_Duration_Minutes'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
IQR


# In[277]:


df['Trip_Duration_Minutes'].fillna(df['Trip_Duration_Minutes'].mean(), inplace=True)


# In[278]:


df['Trip_Duration_Minutes'].count()


# ### Outliers

# In[279]:


# Finding outliers through Box and whisker plot

sns.boxplot(data=df['Trip_Duration_Minutes'], color='lightgreen')
plt.title('Boxplot')
plt.show()


# In[280]:


# Trip_Price


# In[281]:


df['Trip_Price'].fillna(df['Trip_Price'].median(), inplace=True)


# In[282]:


# Handling Outliers

df['Trip_Price'] = df['Trip_Price'].apply(lambda x: 50 if x > 117 else x)


# ## Training and Testing

# In[297]:


X = df.drop(columns=['Trip_Price'])
y = df[['Trip_Price']]


# In[298]:


scaler = MinMaxScaler()         
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[300]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[301]:




model = LinearRegression()


# In[302]:



model.fit(X_train,y_train)


# In[303]:


y_pred = model.predict(X_test)


# In[304]:


mse = mean_squared_error(y_test,y_pred)


# In[305]:



mse


# In[306]:


np.sqrt(mse)


# In[309]:


# Scatterplot for actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='lightgreen', alpha=0.7, label='Predicted vs Actual')

# Plot a perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')

# Labels and title
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




