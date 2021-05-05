#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: clustering.py
# SPECIFICATION: Use k-means and agglomerative clustering to find the homogenity score for dataset
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.to_numpy()

    #run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
k_values = []
k_max = 0
max_coefficient = 0
for k in range(2, 20+1):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    coef = silhouette_score(X_training, kmeans.labels_)
    k_values.append([k,coef])
    if coef > max_coefficient:
        max_coefficient = coef
        k_max = k


#for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)#find which k maximizes the silhouette_coefficient
#--> add your Python code here


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here

df2 = pd.DataFrame(k_values, columns=['k', 'coefficient'])
df2 = df2.set_index('k')

plt.plot(df2, label='Silhouette Coefficient')
plt.legend()
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here

df = pd.read_csv('testing_data.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

labels = np.array(df.values).reshape(1, df.shape[0])[0]
# print(labels)
#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=k_max, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
