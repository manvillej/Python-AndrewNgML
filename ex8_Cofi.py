## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
## imports
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ex8helper as helper

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('Loading movie ratings dataset....')

#load data
mat = io.loadmat('./data/ex8_movies.mat')
Y = mat['Y']
R = mat['R']


#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {:.2f} / 5'.format(np.mean(Y[0,R[0,:]])))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y, cmap='Greys', norm=colors.Normalize(vmin=Y.min(),vmax=Y.max()))
plt.xlabel('Movies')
plt.ylabel('Users')
plt.show()

input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
print('Testing the cost function for Collaborative Filtering...\n')

mat = io.loadmat('./data/ex8_movieParams.mat')
X = mat['X']
Theta = mat['Theta']
numUsers = mat['num_users']
numMovies = mat['num_movies']
numFeatures = mat['num_features']

#  Reduce the data set size so that this runs faster
numUsers = 4
numMovies = 5
numFeatures = 3

X = X[0:numMovies,0:numFeatures]
Theta = Theta[0:numUsers,0:numFeatures]
Y = Y[0:numMovies,0:numUsers]
R = R[0:numMovies,0:numUsers]


#  Evaluate cost function
params = np.append(X.flatten(), Theta.flatten())
J = helper.cofiCostFunction(params, Y, R, numUsers, numMovies, numFeatures, 0)

print('Cost at loaded parameters: {:.2f}'.format(J))
print('(this value should be about 22.22)')


input('\nPart 2 completed. Program paused. Press enter to continue: ')

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
print('Checking Gradients (without regularization) ... \n')

#  Check gradients by running checkNNGradients
helper.checkCostFunction()

input('\nPart 3 completed. Program paused. Press enter to continue: ')

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  

#  Evaluate cost function