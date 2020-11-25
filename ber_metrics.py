import numpy as np
from scipy.spatial import distance, KDTree


class BEREstimator:
    def __init__(self, x, y, subgroups=None):
        """
        Initialize BEREstimator class instance.

        Parameters
        ----------
            x: numpy array
                Each row should be one data point, and each column represents a
                feature.

            y: numpy array
                This array should have length equal to the number of rows in
                `x`. Each entry should be 0 or 1.

            subgroups: dict(str -> numpy array)
                String keys represent the category of subgroup (e.g., race,
                gender, etc.)
        """
        self.x = x
        self.y = y
        self.subgroups = subgroups # not currently used

    def mahalanobis_bound(self):
        """
        Calculate the BER upper bound estimate using the Mahalanobis distance between
        instances in class 0 and class 1.

        Equations from Tumer and Ghosh (2003) and also referencing Ryan Holbrook's
        implementations at:
        https://rdrr.io/github/ryanholbrook/bayeserror/src/R/bayeserror.R

        """
        p_1 = self.y.mean()
        p_0 = 1 - p_1
        mu_0 = self.x[self.y == 0, :].mean(axis=0)  # mean vector for class 0 instances
        mu_1 = self.x[self.y == 1, :].mean(axis=0)  # mean vector for class 1 instances
        sigma_0 = np.cov(self.x[self.y == 0, :].T)
        sigma_1 = np.cov(self.x[self.y == 1, :].T)
        sigma_inv = np.linalg.inv(sigma_0 * p_0 + sigma_1 * p_1)
        m_dist = distance.mahalanobis(mu_0, mu_1, sigma_inv) ** 2
        return 2 * p_0 * p_1 / (1 + p_0 * p_1 * m_dist)

    def bhattacharyya_bound(self):
        """
        Calculate the BER upper bound estimate using the Bhattacharrya bound
        between instances in class 0 and class 1.

        Equations from Tumer and Ghosh (2003) and also referencing Ryan Holbrook's
        implementations at:
        https://rdrr.io/github/ryanholbrook/bayeserror/src/R/bayeserror.R

        """
        p_1 = self.y.mean()
        p_0 = 1 - p_1
        mu_0 = self.x[self.y == 0, :].mean(axis=0)  # mean vector for class 0 instances
        mu_1 = self.x[self.y == 1, :].mean(axis=0)  # mean vector for class 1 instances
        sigma_0 = np.cov(self.x[self.y == 0, :].T)
        sigma_1 = np.cov(self.x[self.y == 1, :].T)
        sigma = (sigma_0 + sigma_1) / 2
        first_term = (1/8) * (mu_1 - mu_0).T @ sigma @ (mu_1 - mu_0)
        second_term = (1/2) * np.log(np.linalg.det(sigma) / np.sqrt(np.linalg.det(sigma_0) * np.linalg.det(sigma_1)))
        b_dist = first_term + second_term
        return np.exp(-b_dist) * np.sqrt(p_0 * p_1) # for now, only interested in upper bound

    def nn_bound(self, test_x, test_y):
        """
        Calculate the BER upper bound estimate using the nearest neighbor method.
        Currently only supports 0/1 binary class.

        Equations from Tumer and Ghosh (2003) and also referencing Ryan Holbrook's
        implementations at:
        https://rdrr.io/github/ryanholbrook/bayeserror/src/R/bayeserror.R

        Parameters
        ----------
            x: numpy array
                Each row should be one data point, and each column represents a
                feature. This represents the test set dataset, which will be
                assigned labels.

            y: numpy array
                This array should have length equal to the number of rows in
                `x`. Each entry should be 0 or 1. This represents the true
                labels of the test set dataset.
        """
        tree = KDTree(self.x)
        _, closest_idx = tree.query(test_x)
        predict_y = self.y[closest_idx]
        err = (predict_y != test_y) / len(test_y)
        return err
