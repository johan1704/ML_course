"""
This code is cuurently not working due to kmeans package
that is surely not recognized...Try it later
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#Applying kmeans to the samll dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualizing the small datatset
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='yellow',label='target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='green',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='sensible')
plt.title("Cluster of client")
plt.xlabel("Annual income($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()

    
