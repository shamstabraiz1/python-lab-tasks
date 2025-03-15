#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# In[2]:


df = pd.read_excel("D:\\software engineering\\5th semester\\machine learning\\github files\\python-lab-tasks\\Online Retail.xlsx")
df.head()


# In[3]:



df.info()


# In[4]:



df.describe()


# In[5]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# In[6]:



df.isnull().sum()


# In[7]:


# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


# In[8]:


plt.figure(figsize=(10, 6))
sns.violinplot(data=df[num_cols], inner="quartile")
plt.xticks(rotation=45)
plt.title("Violin Plot of Numerical Columns")
plt.show()


# In[9]:



# Step 1: Handle Missing Values
df = df.dropna(subset=['CustomerID'])  # Drop rows with missing CustomerID
df = df.drop(columns=['Description'])  # Remove unnecessary text column


# In[10]:


# There are some negative values in Quantity and Unit price Handling it

# Step 2: Remove Cancellations (Negative Quantities)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]  # Ensure valid transactions


# In[11]:



# Step 4: Calculate Total Spending per Transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# In[12]:


# Step 5: Aggregate Data by CustomerID for Clustering
customer_data = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',      # Total spending
    'Quantity': 'sum',        # Total items purchased
    'InvoiceNo': 'nunique'    # Number of transactions
}).rename(columns={'InvoiceNo': 'NumTransactions'})


# In[13]:


# Log transform to reduce skewness (adding 1 to avoid log(0))
customer_data[['TotalPrice', 'Quantity']] = np.log1p(customer_data[['TotalPrice', 'Quantity']])


# In[14]:


# Step 7: Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)


# In[15]:


print("Preprocessed Data Shape:", X_scaled.shape)
customer_data.head()


# ## Clustering with Random Value of _K

# In[16]:


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

# In[17]:


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


# In[18]:


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


# ## DB SCAN

# In[19]:


customer_data.head()


# In[21]:


# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps and min_samples may need tuning
customer_data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Analyze DBSCAN results
print("Number of clusters (excluding noise):", len(set(customer_data['DBSCAN_Cluster'])) - 1)
print("Noise points (Cluster -1):", (customer_data['DBSCAN_Cluster'] == -1).sum())

# Plot results
plt.figure(figsize=(7,5))
sns.scatterplot(x=customer_data['TotalPrice'], y=new_customer_data['Quantity'], hue=customer_data['DBSCAN_Cluster'], palette="viridis")
plt.title("Customer Clustering (DBSCAN)")
plt.xlabel("TotalPrice (Scaled)")
plt.ylabel("Quantity (Scaled)")
plt.legend(title="Cluster")
plt.show()

# Count customers in each cluster
print(customer_data['DBSCAN_Cluster'].value_counts())


# In[23]:


from sklearn.metrics import silhouette_score

# Silhouette score (only for K-Means)
kmeans_silhouette = silhouette_score(X_scaled, customer_data['Cluster'])
print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

# Count of points per cluster
print("\nK-Means Cluster Counts:\n", customer_data['Cluster'].value_counts())
print("\nDBSCAN Cluster Counts:\n", customer_data['DBSCAN_Cluster'].value_counts())

# Creating subplots for side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# K-Means clustering visualization
sc = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=customer_data['Cluster'], cmap='viridis', alpha=0.7)
ax1.set_title('K-Means Clustering')
ax1.set_xlabel("TotalPrice (Scaled)")
ax1.set_ylabel("Quantity (Scaled)")
fig.colorbar(sc, ax=ax1, label="Cluster")  # Attach colorbar to ax1

# DBSCAN clustering visualization
sns.scatterplot(x=customer_data['TotalPrice'], y=customer_data['Quantity'], hue=customer_data['DBSCAN_Cluster'], palette="viridis", ax=ax2)
ax2.set_title('DBSCAN Clustering')
ax2.set_xlabel("TotalPrice (Scaled)")
ax2.set_ylabel("Quantity (Scaled)")
ax2.legend(title="Cluster")

# Show the final figure (prevents empty extra plots)
plt.tight_layout()
plt.show()


# In[ ]:


# DBSCAN, on the other hand, identified multiple small clusters (1, 2, and 3) along with a large main cluster (0) and some noise points (-1). This method is better at handling outliers but is sensitive to parameter tuning.
#While K-Means provides clear segmentation, DBSCAN is more flexible for detecting anomal

#K-Means formed 2 distinct clusters, as shown by the yellow and purple regions, with a moderate silhouette score of 0.495, indicating well-separated clusters.

