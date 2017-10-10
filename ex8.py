## Machine Learning Online Class
#  Exercise 8 |a Anomly Detection and Collaborative Filtering
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
#Imports:
import scipy.io as io
import matplotlib.pyplot as plt

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

print('Visualizing example dataset for outlier detection.\n')

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = io.loadmat('./data/ex8data1.mat');
X = mat['X']

#  Visualize the example dataset
plt.scatter(X[:,0], X[:,1], marker='x', color='b', s=5)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

input('Part 1 completed. Program paused. Press enter to continue: ')

## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution, 
#  then compute the probabilities for each of the points and then visualize 
#  both the overall distribution and where each of the points falls in 
#  terms of that distribution.
#

print('Visualizing Gaussian fit...\n')