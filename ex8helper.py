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

	plt.scatter(X[:,0], X[:,1], marker='x', color='b', s=5)

	infinity = np.sum(np.isinf(Z))

	if(infinity==0):
		V = np.arange(20,0,step=-3, dtype=np.float64)
		V = 1/np.power(10,V)
		plt.contour(X1,X2,Z,V,norm=colors.LogNorm(vmin=V.min(),vmax=V.max()),cmap='inferno_r')