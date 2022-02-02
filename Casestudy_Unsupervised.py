#!/usr/bin/env python
# coding: utf-8

# # CASE STUDY ON UNSUPERVISED LEARNING

# # 1. Read the dataset to the python environment.

# In[2]:


#importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


Data=pd.read_csv("Wine.csv")
Data.head()


# In[4]:


Data.info


# In[4]:


Data.describe()


# In[5]:


Data.shape


# In[6]:


Data.dtypes


# In[7]:


Data.isnull().sum()


# In[5]:


X=Data.iloc[:,1:14]
X


# # 2. Try out different clustering models in the wine dataset

# In[6]:


sns.pairplot(X)


# # 
# <br>
# The 2 clustering methods are:
# <br>
# 1) KMeans Clustering
# <br>
# 2) Agglomerative Clustering
# 

# In[7]:


#kmeans clustering
from sklearn.cluster import KMeans
model_Kmeans=KMeans(n_clusters=3)
model_Kmeans.fit(X)
labels=model_Kmeans.predict(X)
print(labels)


# In[9]:


#Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(X)
y_hc


# # 3.Find the optimum number of clusters in each model and create the model with the optimum number of clusters

# # 
# In case of KMeans clustering we use "Elbow Method" to find out the optimum number of clusters and 
# 
# In case of Agglomerative clustering we use "Dendrogram" to find out the optimum number of clusters.

# In[7]:


#finding the optimum number of clusters using "Elbow Method"
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
    


# In[8]:


#kmeans clustering
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
y_kmeans


# In[47]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="green",label="cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="blue",label="cluster 3")
plt.title("Cluster of wine")
plt.xlabel("Wine")
plt.ylabel("Rate")
plt.legend()
plt.show()


# In[41]:


#finding the optimum number of clusters using "Dendrogram"
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Wine")
plt.ylabel("Euclidean Distances")
plt.show()


# In[42]:


#Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(x)
y_hc


# In[43]:


plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="green",label="cluster 2")
plt.title("Cluster of wine")
plt.xlabel("Alcohol")
plt.ylabel("Rate")
plt.legend()
plt.show()

