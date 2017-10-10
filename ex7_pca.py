## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m - completed
#     projectData.m - completed
#     recoverData.m - completed
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
import matplotlib.pyplot as plt
import ex7helper as helper

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('\nVisualizing example dataset for PCA...\n')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
mat = io.loadmat('./data/ex7data1.mat')
X = mat['X']

plt.title('Raw Data')
plt.scatter(X[:,0], X[:,1], marker='o', color='b', s=10)
plt.show()

input('Part 1 completed. Program paused. Press enter to continue: ')


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset...\n');

#  Before running PCA, it is important to first normalize X
Xnorm = helper.featureNormalize(X)

#  Run PCA
[U,S,V] = helper.PCA(Xnorm)


#  Compute mu, the mean of the each feature
mu = np.mean(X, axis=0)

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
plt.title('Data plotted with PCA vectors')
plt.scatter(X[:,0], X[:,1], marker='o', color='b', s=10)
helper.drawLines(mu, mu + 1.5*S[0]*U[0,:], 'r')
helper.drawLines(mu, mu + 1.5*S[1]*U[1,:], 'g')
plt.show()

print('Top eigenvector: ')
print(U[0,:])
print('(you should expect to see -0.707107 -0.707107)\n')

input('Part 2 completed. Program paused. Press enter to continue: ')

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.

print('Dimension reduction on example dataset...\n')

#  Plot the normalized dataset (returned from pca)

plt.scatter(Xnorm[:,0], Xnorm[:,1], marker='o', color='b', s=10)
plt.axis([-4, 3, -4, 3]) #axis square

#  Project the data onto K = 1 dimension
K = 1
Z = helper.projectData(Xnorm, U, K)

print('Projection of the first example: {}'.format(Z[0]))
print('(this value should be about 1.481274)\n')


Xrecovered  = helper.recoverData(Z, U, K)

print('Approximation of the first example: {:.6f} {:.6f}'.format(Xrecovered[0,0], Xrecovered[0,1]))
print('(this value should be about  -1.047419 -1.047419)\n')

plt.title('Data plotted with PCA vectors')
plt.scatter(X[:,0], X[:,1], marker='o', color='b', s=10)

plt.plot(Xrecovered[:,0],Xrecovered[:,1],color='g')

for i in range(Xnorm.shape[0]):
	helper.drawLines(Xnorm[i,:],Xrecovered[i,:],'k')

plt.show()

input('Part 3 completed. Program paused. Press enter to continue: ')

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n')

mat = io.loadmat('./data/ex7faces.mat')
X = mat['X']

#  Display the first 100 faces in the dataset

helper.displayData(X[:100,:])
plt.show()

input('Part 4 completed. Program paused. Press enter to continue: ')

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
print('Running PCA on face dataset.')
print('(this might take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature

Xnorm = helper.featureNormalize(X)

#  Run PCA
[U,S,V] = helper.PCA(Xnorm)

#  Visualize the top 36 eigenvectors found
helper.displayData(U[:,:36].T)
plt.show()

input('Part 5 completed. Program paused. Press enter to continue: ')

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset...')

K = 100
Z = helper.projectData(Xnorm, U, K)
print('The projected data Z has a size of: ', Z.shape)

input('\nPart 6 completed. Program paused. Press enter to continue: ')

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
print('Visualizing the projected (reduced dimension) faces....\n')

Xrecovered  = helper.recoverData(Z, U, K)

# Display normalized data

# Charting
plt.figure(2)
plt.title('Original Data')
helper.displayData(Xnorm[:100, :])

plt.figure(1)
plt.title('Recovered Data')
helper.displayData(Xrecovered[:100, :])

# Display
plt.show()

input('\nPart 7 completed. Program paused. Press enter to continue: ')