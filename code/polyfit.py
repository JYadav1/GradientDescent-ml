## Polynomial Curve Fitting Exercise

import argparse
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
import scaling as scal



# Read data matrix X and labels t from text file.
def read_data(file_name):
    data_file = np.loadtxt(file_name)
    X = data_file[:, 0] # square ft
    t = data_file[:, 1] # price
    return X, t

def scale(X):
    dataset_ones = np.column_stack((np.ones(X.shape[0]), X))
    data_mean, std = scal.mean_std(dataset_ones) 
    data_scaled = scal.standardize(dataset_ones, data_mean, std)
    return data_scaled[:, 1]

# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient_without_regular(X, t, w):
    N = X.shape[0] # N is the number of rows
    X = X.T # transpose X
    grad = (X@(w@(X) - t))/(N) # calculates the gradient descent
    return grad



def train_without_regular(X, t, eta, epochs):
    w_t = np.array([0,0,0,0,0,0]) 
    for e in range(epochs):
        w_tplus1 = w_t - (eta * compute_gradient_without_regular(X, t, w_t))
        w_t = w_tplus1

    return w_t



def compute_gradient_with_regular(X, t, w, l):
    N = X.shape[0] # N is the number of rows
    X = X.T # transpose X
    grad = l * w + (X@(w@(X) - t))/N # calculates the gradient descent
    return grad

def create_x_matrix(X, M):
    X = X[:, np.newaxis]
    X_column = np.power(X, 0)
    for i in range(1, M+1):
        X_column = np.column_stack((X_column, np.power(X, i)))
    return X_column



def train_with_regular(X, t, eta, epochs, ln):
    l = math.exp(ln)
    # data_mean, std = scal.mean_std(X) 
    # data_scaled = scal.standardize(X, data_mean, std)
    w_t = np.array([0,0,0,0,0,0,0,0,0,0]) 
    for e in range(epochs):
        w_tplus1 = w_t - (eta * compute_gradient_with_regular(X, t, w_t, l))
        w_t = w_tplus1

    return w_t





# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
    N = len(X) # Length of X square ft stored in N
    first = (1)/(2*N) # The first part of the equation 1/2N
    second = np.sum((np.dot(X, w) - t)**2) # second part of equation which consists of sum and squaring it
    cost = first * second # The cost function all together to get the actual value
    return cost




##======================= Main program =======================##
parser = argparse.ArgumentParser('Polynomial Curve Fitting')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training data.
X_dataset, t_dataset = read_data(FLAGS.input_data_dir + "/dataset.txt")
X_train, t_train = read_data(FLAGS.input_data_dir + "/train.txt")
X_test, t_test = read_data(FLAGS.input_data_dir + "/test.txt")
X_devel, t_devel = read_data(FLAGS.input_data_dir + "/devel.txt")


# # Plotting  for 4a # #

# plot for dataset, Part A
dataset_ones = np.column_stack((np.ones(X_dataset.shape[0]), X_dataset))
data_mean, std = scal.mean_std(dataset_ones) 
scaled_dataset_x = scal.standardize(dataset_ones, data_mean, std)
plt.figure(0)
plt.title('X_dataset vs t_dataset')
plt.scatter(scaled_dataset_x[:, 1], t_dataset, color = 'blue') # the dataset data
plt.xlabel("scaled dataset x")
plt.ylabel("t_dataset")
plt.savefig('X-t-dataset.png')

# # Part B # #

# # Plotting for 4b # #

# plot for train data
dataset_ones = np.column_stack((np.ones(X_train.shape[0]), X_train))
data_mean, std = scal.mean_std(dataset_ones) 
scaled_train_x = scal.standardize(dataset_ones, data_mean, std)
plt.figure(1)
plt.title('X_train vs t_train')
plt.scatter(scaled_train_x[:, 1], t_train, color = 'blue') # the train data
plt.xlabel("scaled train x")
plt.ylabel("t_train")
plt.savefig('X-t-train.png')

# plot for test data
dataset_ones = np.column_stack((np.ones(X_test.shape[0]), X_test))
data_mean, std = scal.mean_std(dataset_ones) 
scaled_test_x = scal.standardize(dataset_ones, data_mean, std)
plt.figure(2)
plt.title('X_test vs t_test')
plt.scatter(scaled_test_x[:, 1], t_test, color = 'blue') # the test data
plt.xlabel("scaled test x")
plt.ylabel("ttest")
plt.savefig('X-t-test.png')

# plot for devel data
dataset_ones = np.column_stack((np.ones(X_devel.shape[0]), X_devel))
data_mean, std = scal.mean_std(dataset_ones) 
scaled_devel_x = scal.standardize(dataset_ones, data_mean, std)
plt.figure(3)
plt.title('X_devel vs t_devel')
plt.scatter(scaled_devel_x[:, 1], t_devel, color = 'blue') # the devel data
plt.xlabel("scaled devel x")
plt.ylabel("t_devel")
plt.savefig('X-t-devel.png')

# # 4d $ $

x_scaled = scale(X_train)
x_matrix = create_x_matrix(x_scaled, 5)



plt.figure(4)
labels = [.0001, .001, .01]
rates = np.array([.0001, .001, .01])
hues = ['b', 'g', 'r']
for j, r in enumerate(rates):
    epochs = np.array(range(0, 8200, 200))
    jw = np.zeros(len(epochs))
    for i, e in enumerate(epochs):
        weight = train_without_regular(x_matrix, t_train, r, e)
        jw[i] = compute_cost(x_matrix, t_train, weight)

    plt.plot(epochs, jw, color = hues[j], label = labels[j] )

plt.title('jw vs #epochs')
plt.xlabel('jw')
plt.ylabel('#epochs')
leg = plt.legend()
plt.savefig('4d_part1.png')


print("Without Regular")

w = train_without_regular(x_matrix, t_train, 0.01, 50000)
print(w)

print('Training RMSE: %0.5f.' % np.sqrt(2 * compute_cost(x_matrix, t_train, w)))

x_scaled_test = scale(X_test)
x_matrix_test = create_x_matrix(x_scaled_test, 5)

print('Testing RMSE: %0.5f.' % np.sqrt(2 * compute_cost(x_matrix_test, t_test, w)))

x_scaled = scale(X_train)
x_matrix = create_x_matrix(x_scaled, 9)


plt.figure(5)
labels = [.00001, .0001, .001]
rates = np.array([.00001, .0001, .001])
hues = ['b', 'g', 'r']
for j, r in enumerate(rates):
    epochs = np.array(range(0, 10200, 200))
    jw = np.zeros(len(epochs))
    for i, e in enumerate(epochs):
        weight = train_with_regular(x_matrix, t_train, r, e, -10)
        jw[i] = compute_cost(x_matrix, t_train, weight)

    plt.plot(epochs, jw, color = hues[j], label = labels[j] )

plt.title('jw vs #epochs')
plt.xlabel('jw')
plt.ylabel('#epochs')
leg = plt.legend()
plt.savefig('4d_part2.png')


print('With Regular')

w = train_with_regular(x_matrix, t_train, 0.001, 150000, -10)
print(w)

print('Training RMSE: %0.5f.' % np.sqrt(2 * compute_cost(x_matrix, t_train, w)))

x_scaled_test = scale(X_test)
x_matrix_test = create_x_matrix(x_scaled_test, 9)

print('Testing RMSE: %0.5f.' % np.sqrt(2 * compute_cost(x_matrix_test, t_test, w)))


