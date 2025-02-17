#!/usr/bin/env python
# coding: utf-8

# # Lab 02 - NumPy Exercises 
# 
# 

# #### Import NumPy as np

# In[5]:


import numpy as np
import random


# #### Create an array of 10 zeros 

# In[30]:


np.zeros(10)


# #### Create an array of 10 ones

# In[32]:


np.ones(10)


# #### Create an array of 10 fives

# In[38]:


np.ones(10)*5


# #### Create an array of the integers from 10 to 50

# In[43]:


array =np.arange(10,51)
array


# #### Create an array of all the even integers from 10 to 50

# In[52]:


array =np.arange(10,51,2)
array


# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[49]:


matrix=np.arange(0,9).reshape(3,3)
matrix


# #### Create a 3x3 identity matrix

# In[5]:


i_matrix=np.eye(3)
i_matrix


# #### Use NumPy to generate a random number between 0 and 1

# In[18]:


array=random.random()   #rondom values in float will also generate with the word uniform #
array


# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[6]:


random_number=np.random.standard_normal(25)
print(random_number)


# #### Create the following matrix:

# In[10]:


matrix=np.array([[1,2,3],[4,5,6],[7,8,9]])
matrix


# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[11]:


array=np.linspace(0,1,20)
print(array)


# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[12]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[39]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[19]:


mat[2:,1:]


# In[29]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[20]:


mat[3,4]


# In[30]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[21]:


mat[:3,1:2]


# In[31]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[22]:


mat[4]


# In[32]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[24]:


mat[3:,:5]


# ### Now do the following

# #### Get the sum of all the values in mat

# In[25]:


np.sum(mat)


# #### Get the standard deviation of the values in mat

# In[27]:


np.std(mat)


# # Great Job!

# In[ ]:




