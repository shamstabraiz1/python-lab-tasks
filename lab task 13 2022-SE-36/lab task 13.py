#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np


# In[2]:


def step_function(x):
    return 1 if x >= 0 else 0


# In[3]:


# Perceptron Training Function
def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    num_features = X.shape[1]  # Number of input features (columns)
    weights = np.zeros(num_features + 1)  # Initialize weights (including bias) with 0
    print("Initial Weights:", weights)

    # Training Loop
    for epoch in range(epochs):
        total_error = 0  # Track errors in this epoch
        for i in range(len(X)):
            x_with_bias = np.insert(X[i], 0, 1)  # Add bias term (1 at index 0)
            weighted_sum = np.dot(weights, x_with_bias)  # Compute weighted sum
            y_pred = step_function(weighted_sum)  # Apply Step Activation Function
            error = y[i] - y_pred  # Compute error
            total_error += abs(error)  # Accumulate total error

            # Update Weights: w = w + η(y_true - y_pred) * x
            weights += learning_rate * error * x_with_bias

        print(f"Epoch {epoch+1}: Weights = {weights}, Total Error = {total_error}")

        if total_error == 0:
            break  # Stop early if no errors

    return weights  # Return trained weights


# In[4]:


# Prediction Function
def predict(X, weights):
    predictions = []
    for i in range(len(X)):
        x_with_bias = np.insert(X[i], 0, 1)  # Add bias term
        weighted_sum = np.dot(weights, x_with_bias)  # Compute weighted sum
        y_pred = step_function(weighted_sum)  # Apply Step Activation Function
        predictions.append(y_pred)
    return predictions


# In[5]:


# OR Gate Dataset
X = np.array([
    [0, 0, 0],
    [0, 0, 1], 
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1] 
])
y = np.array([0, 1, 1, 1, 1, 1, 1, 1]) 


# In[6]:


weights = train_perceptron(X, y)

# Test Perceptron
predictions = predict(X, weights)
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {predictions[i]}, Actual: {y[i]}")


# In[ ]:




