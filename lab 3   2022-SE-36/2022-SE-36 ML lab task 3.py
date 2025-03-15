#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
# ___

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[27]:


import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[29]:


df = pd.read_csv("C:\\Users\\shams\\OneDrive\\Desktop\\Salaries.csv")
df


# ** Check the head of the DataFrame. **

# In[30]:


df.head()


# ** Use the .info() method to find out how many entries there are.**

# In[31]:


df.info()


# **What is the average BasePay ?**

# In[35]:


df['BasePay'].mean()


# ** What is the highest amount of OvertimePay in the dataset ? **

# In[36]:


df['OvertimePay'].max()


# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[38]:


df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']


# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[39]:


df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']


# ** What is the name of highest paid person (including benefits)?**

# In[40]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]


# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[41]:


df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]


# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[42]:


df.groupby('Year')['BasePay'].mean()


# ** How many unique job titles are there? **

# In[43]:


df['JobTitle'].nunique()


# ** What are the top 5 most common jobs? **

# In[44]:


df['JobTitle'].value_counts().head()


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[46]:


df[df['Year'] == 2013]['JobTitle'].value_counts().eq(1).sum()


# # Great Job!

# In[ ]:




