from sklearn.preprocessing import OneHotEncoder
import numpy as np

# create dummy categorical data w 100 rows
data = np.random.choice(["hi", "bye", "sky"], size=100).reshape(100, 1)
# perform one hot encoding on categorical variable - make one new column for each category
enc_nodrop = OneHotEncoder(sparse=False)
X1 = enc_nodrop.fit_transform(data)
print(X1.shape) # note that it has 3 columns
print("Determinant of one-hot encoded covariance matrix:",  np.linalg.det(np.cov(X1.T)))
# determinant: -6.15e-18

# perform one hot encoding but drop one column, because the last column is always a linear combination of the others
enc_drop = OneHotEncoder(sparse=False, drop='first')
X2 = enc_drop.fit_transform(data)
print(X2.shape) # note that it has 2 columns
print("Determinant of one-hot encoded covariance matrix with one column dropped:",  np.linalg.det(np.cov(X2.T)))
# determinant: 0.0375