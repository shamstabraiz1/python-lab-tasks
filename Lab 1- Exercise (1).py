#!/usr/bin/env python
# coding: utf-8

# 
# # Lab 1 2022-SE-36
# 
# The objective of this lab is to introduce students to Python programming and the development environment, using tools like Jupiter Notebook and Google Colab, covering key concepts such as variables, data types, loops, conditionals, functions, data structures. 

# ## Exercises
# 
# Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

# ** What is 7 to the power of 4?**

# In[5]:


7**4


# ** Split this string:**
# 
#     s = "Hi there Sam!"
#     
# **into a list. **

# In[13]:


s = "Hi there Sam!"


# In[14]:


s.split()


# ** Given the variables:**
# 
#     planet = "Earth"
#     diameter = 12742
# 
# ** Use .format() to print the following string: **
# 
#     The diameter of Earth is 12742 kilometers.

# In[15]:


planet = "Earth"
diameter = 12742


# In[16]:


"The diameter of earth is {}".format(diameter)


# ** Given this nested list, use indexing to grab the word "hello" **

# In[6]:


lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]


# In[13]:


lst[3][1][2]


# ** Given this nested dictionary grab the word "hello". Be prepared, this will be annoying/tricky **

# In[52]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[73]:


d["k1"][3]['tricky'][3]['target'][3]


# ** What is the main difference between a tuple and a list? **

# In[23]:


#The difference between tupel and list is that tuple is immutable if once created their will no changes will make and their will
#no string store in tuple. and list is mutable and changeable the changes will made if once created.


# ** Create a function that grabs the email website domain from a string in the form: **
# 
#     user@domain.com
#     
# **So for example, passing "user@domain.com" would return: domain.com**

# In[21]:


def domainGet(name):
    return name.split('@')[1]
    
    


# In[22]:


domainGet('user@domain.com')


# ** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **

# In[29]:


def findDog(str):
    return str.count('dog')



# In[30]:


findDog('Is there a dog here?')


# ** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **

# In[31]:


def countDog(str):
    return str.lower().count('dog')


# In[32]:


countDog('This dog runs faster than the other dog dude!')


# 
# **You are driving a little too fast, and a police officer stops you. Write a function
#   to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
#   If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
#   and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
#   cases. **

# In[8]:


def caught_speeding(speed, is_birthday):
    pass
    if is_birthday:
        speed -= 5
    if speed <= 60:
        return "No Ticket"
    elif 61 <= speed <=80:
        return "Small Ticket"
    else:
        return "Big Ticket"


# In[9]:


caught_speeding(81,True)


# In[10]:


caught_speeding(81,False)


# # Great job!
