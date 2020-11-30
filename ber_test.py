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
print("Mahalanobis bound:", ber.mahalanobis_bound())

# R package output for bhattacharyya bound: 1.87519304680994e-05
print("Bhattacharyya bound:", ber.bhattacharyya_bound())

# R package output for NN bound: 0.166666
print("Nearest neighbor bound:", ber.nn_bound())

# can't really test this b/c the R package is not correctly implemented,
# but we can at least test that the bound is something reasonable
individual_predictions = np.array(
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
).T
print("Mutual information-based correlation ensemble bound:", ber.mi_ensemble_bound(individual_predictions))

# the first bhattacharyya bound was 0; I wanted a larger number to verify
# approximate equality between R package and our package
x = np.random.normal(size=(200, 2))
y = np.random.choice([0, 1], p=[0.25, 0.75], size=(200))
ber = BEREstimator(x, y)
# R package output for bhattacharyya bound: 0.407093513312961 (diff random sample)
print("Bhattacharyya bound for second dataset:", ber.bhattacharyya_bound())
