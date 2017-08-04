def computeCost(x,y,theta):
	import numpy as np
	j = 1/2*np.mean(np.power(np.matmul(x.transpose(),theta)-y,2))

	return j