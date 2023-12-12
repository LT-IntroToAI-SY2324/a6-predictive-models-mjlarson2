import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#imports the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
x = data[["Annual Income", "Spending Score"]]

#standardize the data
scaler = StandardScaler()
standard = scaler.fit_transform(x)

#the value of k has been defined for you
k = 5
model = KMeans(n_clusters=k)

#apply the kmeans algorithm
model.fit(standard)

#get the centroid and label values
centroids = model.cluster_centers_
labels = model.labels_

# unscales the data(?)
# standard = scaler.inverse_transform(standard)
# scaler.inverse_transform(x)

#sets the size of the graph
plt.figure(figsize=(5,4))
 
#use a for loop to plot the data points in each cluster
for i in range(k):
    cluster = standard[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1])

#plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=100, c="red", label="Centroid")
            
#shows the graph
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()