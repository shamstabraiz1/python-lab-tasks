#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ### Link of Dataset

# #### https://drive.google.com/drive/folders/1YgHEtqaN0cWbgvoz1CkeNWTpCOMf81Qi?usp=sharing

# In[8]:


# Load MNIST datasets and subset
train_df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\mnist_train.csv").sample(n=10000, random_state=42)
test_df = pd.read_csv("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\mnist_test.csv").sample(n=2000, random_state=42)


# In[22]:


train_df.head()


# In[23]:


test_df.head()


# In[9]:


# Split features and labels
X_train = train_df.drop(columns=['label']).astype('float32') / 255.0
y_train = train_df['label']
X_test = test_df.drop(columns=['label']).astype('float32') / 255.0
y_test = test_df['label']


# In[10]:


def plot_confusion_matrix(true_labels, predicted_labels, model_name):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()


# ##  1. Logistic Regression

# 
# log_reg = LogisticRegression(max_iter=10, solver='saga', multi_class='multinomial')
# log_reg.fit(X_train, y_train)
# log_pred = log_reg.predict(X_test)
# log_acc = accuracy_score(y_test, log_pred)
# print(f"Logistic Regression Accuracy: {log_acc:.4f}")

# In[18]:


# Plot confusion matrix for Logistic Regression
plot_confusion_matrix(y_test, log_pred, "Logistic Regression")


# ## 2. K-Nearest Neighbour "KNN"

# In[12]:


# 3. KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_acc:.4f}")


# In[19]:


# Plot confusion matrix for KNN
plot_confusion_matrix(y_test, knn_pred, "KNN")


# ##  3. Neural Network (MLP)

# In[13]:


# 4. Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50, activation="relu")
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print(f"Neural Network Accuracy: {mlp_acc:.4f}")


# In[20]:


# Plot confusion matrix for Neural Network
plot_confusion_matrix(y_test, mlp_pred, "Neural Network")


# ### comparison

# In[15]:


# Simple comparison
models = {
    "Logistic Regression": log_acc,
    "KNN": knn_acc,
    "Neural Network": mlp_acc
}
best_model_name = max(models, key=models.get)
best_model_acc = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_acc:.4f}")


# In[17]:


# Show detailed report and confusion matrix for the best model
best_pred = {"Logistic Regression": log_pred, 
             "KNN": knn_pred, "Neural Network": mlp_pred}[best_model_name]
print(f"\n{best_model_name} Report:\n", classification_report(y_test, best_pred))
plot_confusion_matrix(y_test, best_pred, best_model_name)


# In[24]:


# Save the best model
best_model = {"Logistic Regression": log_reg,  
              "KNN": knn, "Neural Network": mlp}[best_model_name]
joblib.dump(best_model, "top_model.pkl")
print("Top model saved as 'top_model.pkl'")


# In[ ]:




