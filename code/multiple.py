## Multiple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt

import scaling as scal


# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data_file = np.loadtxt(file_name)
  X = data_file[:, 0:3] # square ft
  t = (data_file[:, 3])/100 # price

  return X, t


# Implement gradient descent algorithm to compute w = [w0, w1, ..., w_M].
def train(X, t, eta, epochs):

#  YOUR CODE here:
  data_mean, std = scal.mean_std(X) 
  data_scaled = scal.standardize(X, data_mean, std)
  w_t = np.array([0,0,0,0]) 
  for e in range(epochs):
    w_tplus1 = w_t - (eta * compute_gradient(data_scaled, t, w_t))
    w_t = w_tplus1

  return w_t




# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  data_mean, std = scal.mean_std(X) 
  X = scal.standardize(X, data_mean, std)
  N = len(X) # Length of X square ft stored in N
  first = (1)/(2*N) # The first part of the equation 1/2N
  second = np.sum((np.dot(X, w) - t)**2) # second part of equation which consists of sum and squaring it
  cost = first * second # The cost function all together to get the actual value
  return cost

# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):
  
#  YOUR CODE here:
  N = X.shape[0] # N is the number of rows
  X = X.T # transpose X
  grad = (X@(w@(X) - t))/(N) # calculates the gradient descent
  return grad



##======================= Main program =======================##
parser = argparse.ArgumentParser('Multiple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/multiple',
                    help='Directory for the multiple regression houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

#  YOUR CODE here: 
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std comptued over *training*.


# Standardize the data
x_train_biased = np.column_stack((np.ones(Xtrain.shape[0]), Xtrain))
x_test_biased = np.column_stack((np.ones(Xtest.shape[0]), Xtest))

# call the train function
wepoch = train(x_train_biased, ttrain, 0.1, 200 )



# print weights
print('Params: ', wepoch)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % np.sqrt(2 * compute_cost(x_train_biased, ttrain, wepoch)))
print('Training cost: %0.2f.' % compute_cost(x_train_biased, ttrain, wepoch))


# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % np.sqrt(2 * compute_cost(x_test_biased, ttest, wepoch)))
print('Test cost: %0.2f.' % compute_cost(x_test_biased, ttest, wepoch))


# compute train 
costs = [] # create an empty list
epochs = list(range(0, 210, 10)) # create a varibale epochs to get the range in 10 increments

for j in range(0, 210, 10):
  weights = train(x_train_biased, ttrain, 0.1, j)
  costs.append(compute_cost(x_train_biased, ttrain, weights))


# computing test
costs_test_data = [] # create an empty list
epochs_test_data = list(range(0, 210, 10)) # create a varibale to get the range in 10 increments

for j in range(0, 210, 10):
  weights = train(x_train_biased, ttrain, 0.1, j)
  costs_test_data.append(compute_cost(x_test_biased, ttest, weights))



plt.figure(1)
plt.title("j(w) Vs #epochs")
plt.plot(epochs, costs, marker = 'o', color = 'blue', label = 'Train') # the train data
plt.plot(epochs_test_data, costs_test_data, marker = '^', color = 'limegreen', label = 'Test') # the test data
plt.xlabel("#epochs")
plt.ylabel("j(w)")

leg1 = plt.legend()

plt.savefig('multiple.png')
  
