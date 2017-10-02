import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import ex2helper as helper2

def plotData(X,y):
	helper2.plotData(X,y)

def svmTrain(X, y, C, func, **kwargs):
	maxPasses = kwargs.pop('maxPasses', 5)
	tol = kwargs.pop('tol', .001)
	m = X.shape[0]
	n = X.shape[1]

	# Map 0 to -1
	y[y==0]=-1

	

