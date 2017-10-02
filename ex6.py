## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m - complete
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Imports:
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

import ex6helper as helper

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#
print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment

mat = io.loadmat('./data/ex6data1.mat')
X = mat['X']
X = np.insert(X,0,np.ones(X.shape[0]),axis=1) #addbias
y = mat['y']

helper.plotData(X,y)
plt.show()

input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

print('\nTraining Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1

#model = helper.svmTrain(X, y, C, @linearKernel, 1e-3, 20)