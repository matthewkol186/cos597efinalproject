from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split

class Ensemble:
    def __init__(self, version, params):
        # versions: 0 = mlp bagging, 1 = random forest, 2 = logreg bagging, 3 = svm bagging, 4 = rbf bagging, 5 = ultimate with 0+1+2+3+4, 6 is 5 without 4 for speed
        # 7 is 0+2+3, so no rf

        self.version = version

        if 'rf_num' not in params.keys():
            params['rf_num'] = 100
        if 'lr_num' not in params.keys():
            params['lr_num'] = 10
        if 'mlp_num' not in params.keys():
            params['mlp_num'] = 10
        if 'svm_num' not in params.keys():
            params['svm_num'] = 10
        if 'rbf_num' not in params.keys():
            params['rbf_num'] = 3
        self.params = params

        if self.version == 0:
            self.model = BaggingClassifier(base_estimator=MLPClassifier(max_iter=300), n_estimators=self.params['mlp_num'])
        elif self.version == 1:
            self.model = RandomForestClassifier(n_estimators=self.params['rf_num'], class_weight='balanced')
        elif self.version == 2:
            self.model = BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=self.params['lr_num'])
        elif self.version == 3:
            self.model = BaggingClassifier(base_estimator=SVC(class_weight='balanced', probability=True), n_estimators=self.params['svm_num'])
        elif self.version == 4:
            kernel = 1.0 * RBF(1.0)
            self.model = BaggingClassifier(base_estimator=GaussianProcessClassifier(kernel=kernel), n_estimators=self.params['rbf_num'])
        elif self.version == 5:
            #self.model = StackingClassifier(estimators=[BaggingClassifier(base_estimator=MLPClassifier(max_iter=300), n_estimators=self.params['mlp_num']), 
            #        RandomForestClassifier(n_estimators=self.params['rf_num'], class_weight='balanced'),
            #        BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=self.params['lr_num']),
            #        BaggingClassifier(base_estimator=SVC(class_weight='balanced'), n_estimators=self.params['svm_num'])])
            kernel = 1.0 * RBF(1.0)
            self.model = [BaggingClassifier(base_estimator=MLPClassifier(max_iter=300), n_estimators=self.params['mlp_num']), 
                    RandomForestClassifier(n_estimators=self.params['rf_num'], class_weight='balanced'),
                    BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=self.params['lr_num']),
                    BaggingClassifier(base_estimator=SVC(class_weight='balanced', probability=True), n_estimators=self.params['svm_num']),
                    BaggingClassifier(base_estimator=GaussianProcessClassifier(kernel=kernel), n_estimators=self.params['rbf_num'])]
            self.combiner = LogisticRegression(class_weight='balanced')
        elif self.version == 6:
            self.model = [BaggingClassifier(base_estimator=MLPClassifier(max_iter=300), n_estimators=self.params['mlp_num']), 
                    RandomForestClassifier(n_estimators=self.params['rf_num'], class_weight='balanced'),
                    BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=self.params['lr_num']),
                    BaggingClassifier(base_estimator=SVC(class_weight='balanced', probability=True), n_estimators=self.params['svm_num'])]
            self.combiner = LogisticRegression(class_weight='balanced')
        elif self.version == 7:
            self.model = [BaggingClassifier(base_estimator=MLPClassifier(max_iter=300), n_estimators=self.params['mlp_num']), 
                    BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=self.params['lr_num']),
                    BaggingClassifier(base_estimator=SVC(class_weight='balanced', probability=True), n_estimators=self.params['svm_num'])]
            self.combiner = LogisticRegression(class_weight='balanced')
        else:
            assert NotImplementedError

    def train(self, X, y):
        if self.version in [5, 6]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

            for i, model in enumerate(self.model):
                model.fit(X_train, y_train)
                print("Fitted model type {0} of {1}".format(i, len(self.model)))
            predictions = []
            for i, model in enumerate(self.model):
                predictions.append(model.predict(X_test))

            self.combiner.fit(np.array(predictions).T, y_test)
        elif self.version in [0, 1, 2, 3, 4]:
            self.model.fit(X, y)

    # returns [num samples x num classifiers in ensemble]
    def test(self, X):
        predictions = []
        if self.version in [0, 1, 2, 3, 4]:
            for estimator in self.model.estimators_:
                predictions.append(estimator.predict(X))
        elif self.version in [5, 6]:
            for model in self.model:
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(X))

        return np.array(predictions).T

    def test_proba(self, X):
        predictions = []
        if self.version in [0, 1, 2, 3, 4]:
            for estimator in self.model.estimators_:
                predictions.append(estimator.predict_proba(X))
        elif self.version in [5, 6]:
            for model in self.model:
                for estimator in model.estimators_:
                    predictions.append(estimator.predict_proba(X))

        return np.array(predictions).T

    # returns one prediction per sample of aggregated ensembles
    def test_agg(self, X, disagg=False):
        if self.version in [0, 1, 2, 3, 4]:
            return self.model.predict(X)
        else:
            predictions = []
            for i, model in enumerate(self.model):
                predictions.append(model.predict(X))

            if disagg:
                return np.array(predictions).T, self.combiner.predict(np.array(predictions).T)
            return self.combiner.predict(np.array(predictions).T)




