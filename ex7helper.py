import numpy as np
import matplotlib.pyplot as plt

def findClosestCentroid(X, centroids):
	#FINDCLOSESTCENTROIDS computes the centroid memberships for every example
	#   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
	#   in idx for a dataset X where each row is a single example. idx = m x 1 
	#   vector of centroid assignments (i.e. each entry in range [1..K])
	#

	#stack centroids so that X and centroid can be subtracted
	centroidMatrix = np.tile(centroids,(X.shape[0],1,1))

	#stack X so that X and centroid can be subtracted
	Xmatrix = np.tile(X,(centroids.shape[0],1,1))

	#swap axes so Xmatrix and centroidMatrix have matching axes
	Xmatrix = np.swapaxes(Xmatrix,0,1)

	'''
	How to check the structure of Xmatrix
	for i in range(3):
		print(Xmatrix[:3, i, :].flatten())
	print(X[:3, :].flatten())
	'''

	#get distance of each value
	results = np.power(Xmatrix-centroidMatrix,2)
	#sum difference
	results = np.sum(results,axis=2)
	#get position of minimum
	results = np.argmin(results,axis=1)
	
	return results

def computeCentroids(X, idx, k):
	#COMPUTECENTROIDS returns the new centroids by computing the means of the 
	#data points assigned to each centroid.
	#   centroids = computeCentroids(X, idx, K) returns the new centroids by 
	#   computing the means of the data points assigned to each centroid. It is
	#   given a dataset X where each row is a single data point, a vector
	#   idx of centroid assignments (i.e. each entry in range [1..K]) for each
	#   example, and K, the number of centroids. You should return a matrix
	#   centroids, where each row of centroids is the mean of the data points
	#   assigned to it.
	#
	centroids = np.zeros((k,X.shape[1]))

	for i in range(k):
		centroids[i,:]= np.mean(X[idx==i,:],axis=0)

	return centroids
