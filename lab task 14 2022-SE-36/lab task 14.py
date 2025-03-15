#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


# In[2]:


# Step 1: Load Dataset
iris = load_iris()
# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[4]:


X, y = iris.data, iris.target  # Features and Labels
X


# In[5]:


y


# In[6]:


# Step 2: Split Data into Training & Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Step 4: Define the MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(10, 5),  # Two hidden layers (10 neurons, 5 neurons)
                    activation='relu',  # ReLU Activation
                    solver='adam',            #optimizer
                    max_iter=500,               # Max Iterations
                    random_state=42)


# In[8]:


mlp


# In[9]:


# Step 5: Train the Model
mlp.fit(X_train, y_train)


# In[10]:


# Step 6: Make Predictions
y_pred = mlp.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[11]:


# Predict probabilities
probs = mlp.predict_proba(X_test)  # Softmax probabilities
print(probs[:5][1])  # Display first 5 predictions


# In[12]:


y_pred


# In[ ]:




