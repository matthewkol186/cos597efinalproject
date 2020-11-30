import numpy as np
from scipy.spatial import distance, KDTree
from pyitlib import discrete_random_variable as drv


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

    def nn_bound(self):
        """
        Calculate the BER upper bound estimate using the nearest neighbor method.
        Currently only supports 0/1 binary class.

        Equations from Tumer and Ghosh (2003) and also referencing Ryan Holbrook's
        implementations at:
        https://rdrr.io/github/ryanholbrook/bayeserror/src/R/bayeserror.R
        """
        tree = KDTree(self.x)
        # we know the closest will be the data point itself, so return
        # 2 nearest neighbors
        _, closest_idx = tree.query(self.x, k=2)
        closest_idx = closest_idx[:, 1].reshape(-1)
        predict_y = self.y[closest_idx]
        err = (predict_y != self.y).sum() / len(self.y)
        return err

    def mi_ensemble_bound(self, individual_predictions):
        """
        Estimate the BER using the Mutual Information-Based Correlation in
        Tumer and Ghosh (2003).

        Parameters
        ----------
            individual_predictions: numpy array
                The dimensions of this array should be |M| by |E|, where
                |M| is the number of labeled data points and |E| is the number
                of individual classifiers. Each element should be a probability
                (not a 0/1 prediction).)
        """
        avg_predictor = individual_predictions.mean(axis=1).round()
        individual_predictions = individual_predictions.round() # deal with 0/1 predictions
        N = individual_predictions.shape[1]  # number of classifiers in ensemble
        labels = np.repeat(self.y.reshape(-1, 1), N, axis=1)
        accs = np.equal(individual_predictions, labels).mean(axis=0) # mean accuracy for each classifier
        mean_err = 1 - accs.mean() # mean err for all classifiers
        ensemble_err = 1 - (self.y == avg_predictor).mean() # mean err for ensemble classifier

        # calculate average mutual information between each individual classifier's
        # predictions and the ensemble predictor
        ami = drv.information_mutual(
            individual_predictions.T,
            avg_predictor.reshape(1, -1),
            base=np.e,
            cartesian_product=True
        ).mean()
        # total entropy in the individual classifiers
        total_entropy = drv.entropy_joint(individual_predictions.T, base=np.e)
        # delta is the normalized ami
        delta = ami / total_entropy
        assert delta >= 0
        assert delta <= 1
        # formula from Tumer and Ghosh
        be = (N * ensemble_err - ((N - 1) * delta + 1) * mean_err ) / ((N - 1) * (1 - delta))
        return be

