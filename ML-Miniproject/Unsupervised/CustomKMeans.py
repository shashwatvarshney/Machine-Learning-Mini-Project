# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:35:57 2019

@author: shashwat
"""
#Kmeans Clustering
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Generate random data and center it to the three centers
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3
X = np.concatenate((data_1, data_2, data_3), axis = 0)
# =============================================================================
# X = np.array([[1,2],
#               [1.5,1.8],
#               [5,8],
#               [8,8],
#               [1,0.6],
#               [9,11]])
# =============================================================================
colors = ["g","r","c","b","k"]
plt.scatter(X[:,0],X[:,1],s=150)
plt.show()
#class custom KMeans Clustering
class K_Means:
    #tol is tolerance and max_iteration is maximum iterations
    def __init__(self, k=3, tol=0.0001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    
    
    def fit(self,data):
        
        self.centroids = {}
        #initializing centroids into first two data points
        for i in range(self.k):
            self.centroids[i] = data[i]
         # dict which contains centroids as key and data points as values   
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            #calculating distances from centroids to all data points
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                #index value is centroid 
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            #finding  centroid
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                #if any of centroids move more than tolerance then not optimized
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            for c in self.centroids:
                color = colors[c]
                for f in self.classifications[c]:
                    plt.scatter(f[0],f[1],marker="x",color=color,s=150,linewidth=4)
            for c in self.centroids:
                plt.scatter(self.centroids[c][0],self.centroids[c][1],marker="o",color="b",s=150,linewidth=5)
            plt.show()
            print(self.centroids[0])
            print(self.centroids[1])
            print(self.centroids[2])

            if optimized:
                break

    def predict(self,data):
                    #calculating distances from centroids to all data point
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
      
clf = K_Means()
clf.fit(X)
    
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=150,linewidth=5)
        
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color="b",s=150,linewidths=5)

plt.show()

