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
import scipy.optimize as op

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
print("Checking Cost function with Regularization...\n")

#  Evaluate cost function
J = helper.cofiCostFunction(params, Y, R, numUsers, numMovies, numFeatures, 1.5)

print('Cost at loaded parameters(lambda = 1.5): {:.2f}'.format(J))
print('(this value should be about 31.34)')     

input('\nPart 4 completed. Program paused. Press enter to continue: ')  

## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#

print('Checking Gradients (with regularization) ... \n')

#  Check gradients by running checkCostFunction
helper.checkCostFunction(1.5)

input('\nPart 5 completed. Program paused. Press enter to continue: ') 

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
# 
print('New user ratings...\n')

titleToId = helper.loadMovieList()
idToTitle = {v: k for k, v in titleToId.items()}

# initialize my ratings
myRatings = np.zeros(1682)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
myRatings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
myRatings[97] = 2


# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
myRatings[6] = 3
myRatings[11] = 5
myRatings[53] = 4
myRatings[63] = 5
myRatings[65] = 3
myRatings[68] = 5
myRatings[182] = 4
myRatings[225] = 5
myRatings[354] = 5

print('User ratings...')

for i in range(myRatings.shape[0]):
	if myRatings[i] > 0:
		print('Rated {} for {}'.format(myRatings[i], idToTitle[i]))

input('\nPart 6 completed. Program paused. Press enter to continue: ') 

## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#

print('Training collaborative filtering...\n')

#  Load data
mat = io.loadmat('./data/ex8_movies.mat')
Y = mat['Y']
R = mat['R']
myRatings = myRatings[:, np.newaxis]

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i
#
#  Add our own ratings to the data matrix
Y = np.concatenate((myRatings, Y), axis=1)
R = np.concatenate(((myRatings>0).astype(int), R), axis=1)

# normalize values
[Ymean, Ynorm] = helper.normalizeRatings(Y, R)

#  Useful Values
numUsers = Y.shape[1]
numMovies = Y.shape[0]
numFeatures = 10

# Set Initial Parameters (Theta, X)
X = np.random.standard_normal((numMovies,numFeatures))
Theta = np.random.standard_normal((numUsers,numFeatures))

initialParams = np.append(X.flatten(), Theta.flatten())

# Set options for fmincg

# Set Regularization
lambdaVal = 10

results = op.minimize(fun=helper.cofiCostFunction, x0=initialParams, args=(Ynorm, R, numUsers, numMovies, numFeatures, lambdaVal), method='TNC', jac=helper.cofiGradFunction, options={'disp': True})
#% Unfold the returned theta back into U and W
#X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
#Theta = reshape(theta(num_movies*num_features+1:end), ...
#               num_users, num_features);


print(results)
params = results.x

X = np.reshape(params[0:X.size],X.shape)
Theta = np.reshape(params[X.size:],Theta.shape)

print('Recommender system learning completed.\n')

input('\nPart 7 completed. Program paused. Press enter to continue: ') 

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
print('Making a prediction...\n')

p = np.matmul(X,Theta.T)
myPredictions = p[:,0] + Ymean

titleToId = helper.loadMovieList()
idToTitle = {v: k for k, v in titleToId.items()}

ix = np.flip(np.argsort(myPredictions),0)
print('Top recommendations for you:')
for i in range(10):
    j = ix[i]
    print('Predicting rating {:.1f} for movie {}'.format(myPredictions[j], idToTitle[j]))

print('\nOriginal ratings provided:')
for i in range(myRatings.size):
    if(myRatings[i] > 0):
    	print('Rated {:.1f} for {}'.format(float(myRatings[i]), idToTitle[i]))

input('\nPart 8 completed. Program completed. Press enter to exit: ')
