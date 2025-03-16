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


df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\Dataset .csv\\Dataset .csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# ### Top 10 Cities

# In[5]:


plt.figure(figsize=(12, 6))
top_cities = df["City"].value_counts().nlargest(20)
sns.barplot(x=top_cities.index, y=top_cities.values, palette="viridis")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.title("Top 10 Cities with the Most Restaurants")
plt.xticks(rotation=45)
plt.show()


# ### Pie Chart: Cuisine Distribution

# In[6]:


plt.figure(figsize=(14, 8))
top_cuisines = df["Cuisines"].value_counts().nlargest(15)
plt.pie(top_cuisines, labels=top_cuisines.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Top 10 Cuisines Distribution")
plt.show()


# ### Satter Plot: Restaurant Locations

# In[7]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Longitude', y='Latitude', hue='Aggregate rating', data=df)
plt.title('Restaurant Locations with Rating Encoding')
plt.show()


# ### Box Plot: Price Range vs. Aggregate Ratin

# In[8]:


plt.figure(figsize=(7,5))
sns.boxplot(x=df["Price range"], y=df["Aggregate rating"], palette="coolwarm")
plt.xlabel("Price Range")
plt.ylabel("Aggregate Rating")
plt.title("Price Range vs. Aggregate Rating")
plt.show()


# ### Scatter Plot: Aggregate Rating vs. Number of Votes

# In[10]:


plt.figure(figsize=(7,5))
sns.scatterplot(x=df["Aggregate rating"], y=df["Votes"], alpha=0.5, color="blue")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Votes")
plt.title("Aggregate Rating vs. Number of Votes")
plt.show()


# ###  Count Plot: Rating Categories

# In[11]:


plt.figure(figsize=(7, 5))
sns.countplot(x=df['Rating text'])
plt.title("Count of Rating Categories")
plt.xlabel("Rating Category")
plt.ylabel("Count")
plt.show()


# ### Bar Plot Top 10 Most Rated Restaurants

# In[12]:


plt.figure(figsize=(10,5))
top_rated = df[['Restaurant Name', 'Votes']].sort_values(by='Votes', ascending=False).head(10)
sns.barplot(y=top_rated['Restaurant Name'], x=top_rated['Votes'], palette='coolwarm')
plt.xlabel('Votes')
plt.ylabel('Restaurant')
plt.title('Top 10 Most Rated Restaurants')
plt.show()


# ### Stacked Bar Chart: Table Availability vs. Online Booking

# In[13]:


table_online = df.groupby(['Has Table booking', 'Has Online delivery']).size().unstack()
table_online.plot(kind="bar", stacked=True, figsize=(6, 4))
plt.title("Table Reservation vs. Online Booking")
plt.xlabel("Has Table")
plt.ylabel("Count")
plt.show()


# ### Box Plot: Has Table Booking vs. Ratings

# In[14]:


plt.figure(figsize=(6,4))
sns.boxplot(x=df['Has Table booking'], y=df['Aggregate rating'])
plt.xlabel('Has Table Booking')
plt.ylabel('Ratings')
plt.title('Table Booking vs Ratings')
plt.show()


# ### Bar Plot Top 10 Localities

# In[17]:


plt.figure(figsize=(8,4))
df['Locality'].value_counts().head(10).plot(kind='bar', color='yellow')
plt.xlabel('Locality')
plt.ylabel('Number of Restaurants')
plt.title('Top 10 Localities with Most Restaurants')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




