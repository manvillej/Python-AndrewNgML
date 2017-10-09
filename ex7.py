## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m - completed
#     findClosestCentroids.m - completed
#     kMeansInitCentroids.m - completed
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
## Imports
import numpy as np
import scipy.io as io
import ex7helper as helper
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you should complete the code in the findClosestCentroids function. 
#

print('Finding closest centroids.\n')

# Load an example dataset that we will be using
mat = io.loadmat('./data/ex7data2.mat')
X = mat['X']


# Select an initial set of centroids
K = 3 # 3 Centroids

initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = helper.findClosestCentroid(X, initialCentroids)


print('Closest centroids for the first 3 examples: ')
print(idx[:3].flatten())
print('(the closest centroids should be 0, 2, 1 respectively)\n')

input('Part 1 completed. Program paused. Press enter to continue: ')

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.')

# Compute means based on the closest centroids found in the previous part.
centroids = helper.computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:')
np.set_printoptions(precision=6)
print(centroids)
print('(the centroids should be:')
print('[[ 2.428301  3.157924]')
print(' [ 5.813503  2.633656]')
print(' [ 7.119387  3.616684]]')

input('\nPart 2 completed. Program paused. Press enter to continue: ')

## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print('\nRunning K-Means clustering on example dataset.')

# Load an example dataset
mat = io.loadmat('./data/ex7data2.mat')
X = mat['X']

# Settings for running K-Means
K = 3;
maxIters = 10;

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
kmeans = KMeans(n_clusters=K).fit(X)
centroids = kmeans.cluster_centers_ 
idx = kmeans.labels_

#charting
plt.figure(2)
plt.title('Data Grouped')
plt.scatter(X[idx==0,0], X[idx==0,1], marker='o', color='b', s=10)
plt.scatter(X[idx==1,0], X[idx==1,1], marker='o', color='r', s=10)
plt.scatter(X[idx==2,0], X[idx==2,1], marker='o', color='g', s=10)
plt.scatter(centroids[0,0], centroids[0,1], marker='x', color='g', s=30)
plt.scatter(centroids[1,0], centroids[1,1], marker='x', color='b', s=30)
plt.scatter(centroids[2,0], centroids[2,1], marker='x', color='r', s=30)

plt.figure(1)
plt.title('Data Not Grouped')
plt.scatter(X[:,0], X[:,1], marker='o', color='k', s=10)
plt.show()

print(np.mean(X[idx==0,:],axis=0))
print(centroids[0])

input('\nPart 3 completed. Program paused. Press enter to continue: ')


## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel onto its closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

print('\nRunning K-Means clustering on pixels from an image.')

X = plt.imread('./data/bird_small.png')
A = X/255 #divide by 255 to reduce values to a range of 0 - 1

#image shape
imgShape = A.shape


# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
newShape = [imgShape[0]*imgShape[1],imgShape[2]]

A = np.reshape(A, newShape)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16 

# Calculate Kmeans clusters
kmeans = KMeans(n_clusters=K, random_state=0).fit(A)
centroids = kmeans.cluster_centers_ 
idx = kmeans.labels_



input('\nPart 4 completed. Program paused. Press enter to continue: ')

## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we set each point to the value of the closest centroid

print('\nApplying K-Means to compress an image.')

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

ARecovered = np.copy(A)


for i in np.unique(idx):
	ARecovered[idx==i,:]=centroids[i]


# Reshape the recovered image into proper dimensions
XRecovered = 255*np.reshape(ARecovered,imgShape)


# Display the original image 
f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(X)
ax1.set_title('Original')
ax2.imshow(XRecovered)
ax2.set_title('Compressed')
plt.show()

input('\nPart 5 completed. Program completed. Press enter to exit: ')