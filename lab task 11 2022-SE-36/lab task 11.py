#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[5]:


df = pd.read_excel("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\Online Retail.xlsx")
df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# In[9]:



df.isnull().sum()


# In[10]:


# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


# In[15]:


plt.figure(figsize=(10, 6))
sns.violinplot(data=df[num_cols], inner="quartile")
plt.xticks(rotation=45)
plt.title("Violin Plot of Numerical Columns")
plt.show()


# In[17]:



# Step 1: Handle Missing Values
df = df.dropna(subset=['CustomerID'])  # Drop rows with missing CustomerID
#df = df.drop(columns=['Description'])  # Remove unnecessary text column


# In[18]:


# There are some negative values in Quantity and Unit price Handling it

# Step 2: Remove Cancellations (Negative Quantities)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]  # Ensure valid transactions


# In[19]:


# Step 4: Calculate Total Spending per Transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# In[20]:


# Step 5: Aggregate Data by CustomerID for Clustering
customer_data = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',      # Total spending
    'Quantity': 'sum',        # Total items purchased
    'InvoiceNo': 'nunique'    # Number of transactions
}).rename(columns={'InvoiceNo': 'NumTransactions'})


# In[21]:


# Log transform to reduce skewness (adding 1 to avoid log(0))
customer_data[['TotalPrice', 'Quantity']] = np.log1p(customer_data[['TotalPrice', 'Quantity']])


# In[22]:


# Step 7: Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)


# In[23]:


print("Preprocessed Data Shape:", X_scaled.shape)
customer_data.head()


# ## Clustering with Random Value of _K

# In[24]:


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 3: Analyze Cluster Characteristics
print(customer_data.groupby('Cluster').mean())

# Step 4 (Optional): Scatter Plot for Visualization (Only for 2D Projection)
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=customer_data['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("TotalPrice (Scaled)")
plt.ylabel("Quantity (Scaled)")
plt.title("Customer Clustering (K-Means)")
plt.colorbar(label="Cluster")
plt.show()


# ## Using Elbow Method

# In[25]:


from sklearn.cluster import KMeans

# Step 1: Find the optimal number of clusters using Elbow Method
inertia = []
K_range = range(1, 11)  # Checking for k = 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method to Find Optimal k")
plt.show()


# In[26]:


# Step 2: Apply K-Means with the chosen k (let's assume k=4 based on elbow method)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 3: Analyze Cluster Characteristics
print(customer_data.groupby('Cluster').mean())

# Step 4 (Optional): Scatter Plot for Visualization (Only for 2D Projection)
plt.figure(figsize=(7,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=customer_data['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("TotalPrice (Scaled)")
plt.ylabel("Quantity (Scaled)")
plt.title("Customer Clustering (K-Means)")
plt.colorbar(label="Cluster")
plt.show()


# In[27]:


from sklearn.metrics import silhouette_score

# Silhouette score (only for K-Means)
kmeans_silhouette = silhouette_score(X_scaled, customer_data['Cluster'])
print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




