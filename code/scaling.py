import numpy as np

# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  X = X[:, 1:]
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])
  ## Your code here.
  mean = X.mean(axis = 0) # The inbuilt mean method to calculate mean with axis = 0
  std = X.std(axis = 0) # The inbuilt std method to calc std with axis = 0

  return mean, std


# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = np.zeros(X.shape)

  ## Your code here.
  S = (X[:, 1:] - mean) / (std) 
  S = np.column_stack((np.ones(X.shape[0]), S))
  return S
