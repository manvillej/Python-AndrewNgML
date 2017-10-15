import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def visualizeFit(X, mean, variance):
	#VISUALIZEFIT Visualize the dataset and its estimated distribution.
	#   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
	#   probability density function of the Gaussian distribution. Each example
	#   has a location (x1, x2) that depends on its feature values.
	space = np.arange(0,35,.5)
	
	[X1, X2] = np.meshgrid(space,space)
	Z = np.array([X1.flatten(), X2.flatten()])

	Z = multivariate_normal.pdf(Z.T, mean=mean, cov=variance)
	Z = np.reshape(Z,X1.shape)

	infinity = np.sum(np.isinf(Z))

	if(infinity==0):
		V = np.arange(20,0,step=-3, dtype=np.float64)
		V = 1/np.power(10,V)
		plt.contour(X1,X2,Z,V,norm=colors.LogNorm(vmin=V.min(),vmax=V.max()),cmap='inferno_r')

def selectThreshold(yval, pval):
	#SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
	#outliers
	#   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
	#   threshold to use for selecting outliers based on the results from a
	#   validation set (pval) and the ground truth (yval).
	#
	bestEpsilon = 0
	bestF1 = 0
	F1 = 0

	#get epsilon values to check
	stepSize = (pval.max() - pval.min())/1000
	rangeOfThresholds = np.arange(pval.min(),pval.max(),stepSize)


	for epsilon in rangeOfThresholds:
    	# Compute the F1 score of choosing epsilon as the
    	# threshold and place the value in F1. The code at the
    	# end of the loop will compare the F1 score for this
    	# choice of epsilon and set it to be the best epsilon if
    	# it is better than the current choice of epsilon.

    	# So, I need to calculate the F1 score
    	# To do that, I need to calculate Precision and Recall
    	# To do that, I need to calculate number of true positives, actual positives, and guessed positives
    	# To do that, I need to calculate the predictions, but that is pretty straight forward
		
		#predictions
		predictions = (pval < epsilon).astype(np.float32)

		#total positives
		positives = float(np.sum(yval==1))

		#guess positives
		guessedPositives = float(np.sum(predictions==1))

		#true positives
		truePositives = float(np.sum((predictions+yval)==2))

		if(guessedPositives==0 or positives==0):
			continue

		#precision
		precision = truePositives/guessedPositives

		#recall
		recall = truePositives/positives

		F1 = 2*(precision*recall)/(precision+recall)
		if(F1 > bestF1):
			bestF1 = F1
			bestEpsilon = epsilon

	return bestF1, bestEpsilon

def cofiCostFunction(params, Y, R, numUsers, numMovies, numFeatures, lambdaVal):
	#COFICOSTFUNC Collaborative filtering cost function
	#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
	#   num_features, lambda) returns the cost and gradient for the
	#   collaborative filtering problem.
	#
	# Unfold the U and W matrices from paramsX
	X = np.reshape(params[0:numMovies*numFeatures],[numMovies,numFeatures])
	Theta = np.reshape(params[numMovies*numFeatures:],[numUsers,numFeatures])

	            
	# You need to return the following values correctly
	J = 0
	XGrad = np.zeros(X.shape)
	ThetaGrad = np.zeros(Theta.shape)

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost function and gradient for collaborative
	#               filtering. Concretely, you should first implement the cost
	#               function (without regularization) and make sure it is
	#               matches our costs. After that, you should implement the 
	#               gradient and use the checkCostFunction routine to check
	#               that the gradient is correct. Finally, you should implement
	#               regularization.
	#
	# Notes: X - num_movies  x num_features matrix of movie features
	#        Theta - num_users  x num_features matrix of user features
	#        Y - num_movies x num_users matrix of user ratings of movies
	#        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
	#            i-th movie was rated by the j-th user
	#
	# You should set the following variables correctly:
	#
	#        X_grad - num_movies x num_features matrix, containing the 
	#                 partial derivatives w.r.t. to each element of X
	#        Theta_grad - num_users x num_features matrix, containing the 
	#                     partial derivatives w.r.t. to each element of Theta
	#
	# returns the cost without regularization
	# added a placeholder matrix to store values
	# of the difference between predictions & actual Y values
	# where there are actual ratings. This is to reduce
	# the computations since it is computed frequently


	placeholder = np.multiply(X.dot(Theta.T) - Y, R)

	# Calculate Cost
	J = np.sum(np.square(placeholder))/2 + lambdaVal/2*(np.sum(np.square(X)) + np.sum(np.square(Theta)))

	return J

def cofiGradFunction(params, Y, R, numUsers, numMovies, numFeatures, lambdaVal):
	#COFICOSTFUNC Collaborative filtering cost function
	#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
	#   num_features, lambda) returns the cost and gradient for the
	#   collaborative filtering problem.
	#
	# Unfold the U and W matrices from paramsX
	X = np.reshape(params[0:numMovies*numFeatures],[numMovies,numFeatures])
	Theta = np.reshape(params[numMovies*numFeatures:],[numUsers,numFeatures])

	            
	# You need to return the following values correctly
	J = 0
	XGrad = np.zeros(X.shape)
	ThetaGrad = np.zeros(Theta.shape)

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost function and gradient for collaborative
	#               filtering. Concretely, you should first implement the cost
	#               function (without regularization) and make sure it is
	#               matches our costs. After that, you should implement the 
	#               gradient and use the checkCostFunction routine to check
	#               that the gradient is correct. Finally, you should implement
	#               regularization.
	#
	# Notes: X - num_movies  x num_features matrix of movie features
	#        Theta - num_users  x num_features matrix of user features
	#        Y - num_movies x num_users matrix of user ratings of movies
	#        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
	#            i-th movie was rated by the j-th user
	#
	# You should set the following variables correctly:
	#
	#        X_grad - num_movies x num_features matrix, containing the 
	#                 partial derivatives w.r.t. to each element of X
	#        Theta_grad - num_users x num_features matrix, containing the 
	#                     partial derivatives w.r.t. to each element of Theta
	#
	# returns the cost without regularization
	# added a placeholder matrix to store values
	# of the difference between predictions & actual Y values
	# where there are actual ratings. This is to reduce
	# the computations since it is computed frequently


	placeholder = np.multiply(X.dot(Theta.T) - Y, R)

	# Calculate X & Theta gradients
	XGrad = placeholder.dot(Theta) + lambdaVal*X
	ThetaGrad = placeholder.T.dot(X) + lambdaVal*Theta

	grad = np.append(XGrad.flatten(), ThetaGrad.flatten())

	return grad