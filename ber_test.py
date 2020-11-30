from ber_metrics import BEREstimator
import numpy as np

# initialize dataset
x1 = (np.arange(16) + 1).reshape((4,4))
x2 = (np.arange(16)+1).reshape((4,4)) ** 2
x3 = (np.arange(16)+1).reshape((4,4)) ** 3
x = np.vstack((x1.T, x2.T, x3.T))
y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# create BEREstimator
ber = BEREstimator(x, y)

# R package output for mahalanobis bound: 0.28749
print(ber.mahalanobis_bound())

# R package output for bhattacharyya bound: 1.87519304680994e-05
print(ber.bhattacharyya_bound())

# R package output for NN bound: 0.166666
print(ber.nn_bound())

x = np.random.normal(size=(200, 2))
y = np.random.choice([0, 1], p=[0.25, 0.75], size=(200))
ber = BEREstimator(x, y)
# R package output for bhattacharyya bound: 0.407093513312961 (diff random sample)
print(ber.bhattacharyya_bound())
