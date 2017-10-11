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
