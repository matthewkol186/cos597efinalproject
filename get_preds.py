import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import pickle
from sklearn import preprocessing
from sklearn import svm
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import time
from ensemble_utils import Ensemble
from ber_metrics import BEREstimator

parser = argparse.ArgumentParser(description='diagnosing some errors')
parser.add_argument('--row', type=int, default=0, help='row in sheets')
parser.add_argument('--dataset', type=str, default='adult', help='adult or compas-arrest or compas-violent')
parser.add_argument('--minority', action='store_true', default=False, help='if set, will look at minority group rather than majority')
args = parser.parse_args()

if args.dataset == 'adult':
    X_train, y_train, X_test, y_test = pickle.load(open('Data/adult_income/processed_data.pkl', 'rb'))
    protected_labels = ['race_Black', 'race_Other', 'race_White', 'sex_Male', 'sex_Female']
elif args.dataset == 'compas-arrest':
    X_train, y_train, X_test, y_test = pickle.load(open('Data/compas/processed_arrest.pkl', 'rb'))
    protected_labels = ['race_is_causasian', 'race_is_african_american', 'race_is_hispanic', 'race_is_other', 'female']
elif args.dataset == 'compas-violent':
    X_train, y_train, X_test, y_test = pickle.load(open('Data/compas/processed_violent.pkl', 'rb'))
    protected_labels = ['race_is_causasian', 'race_is_african_american', 'race_is_hispanic', 'race_is_other', 'female']
else:
    assert NotImplementedError


if args.row == 1:
    X_train = X_train.drop(labels=protected_labels, axis=1)
    X_test = X_test.drop(labels=protected_labels, axis=1)
elif args.row == 3:
    if args.minority:
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['race_Black']==1], y_train[X_train['race_Black']==1]
            X_test, y_test = X_test[X_test['race_Black']==1], y_test[X_test['race_Black']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['race_is_african_american']==1], y_train[X_train['race_is_african_american']==1]
            X_test, y_test = X_test[X_test['race_is_african_american']==1], y_test[X_test['race_is_african_american']==1]
    else:
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['race_White']==1], y_train[X_train['race_White']==1]
            X_test, y_test = X_test[X_test['race_White']==1], y_test[X_test['race_White']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['race_is_causasian']==1], y_train[X_train['race_is_causasian']==1]
            X_test, y_test = X_test[X_test['race_is_causasian']==1], y_test[X_test['race_is_causasian']==1]
    X_train = X_train.drop(labels=protected_labels, axis=1)
    X_test = X_test.drop(labels=protected_labels, axis=1)
elif args.row == 4:
    if args.minority:
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['sex_Female']==1], y_train[X_train['sex_Female']==1]
            X_test, y_test = X_test[X_test['sex_Female']==1], y_test[X_test['sex_Female']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['female']==1], y_train[X_train['female']==1]
            X_test, y_test = X_test[X_test['female']==1], y_test[X_test['female']==1]
    else:
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['sex_Male']==1], y_train[X_train['sex_Male']==1]
            X_test, y_test = X_test[X_test['sex_Male']==1], y_test[X_test['sex_Male']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['female']==0], y_train[X_train['female']==0]
            X_test, y_test = X_test[X_test['female']==0], y_test[X_test['female']==0]
    X_train = X_train.drop(labels=protected_labels, axis=1)
    X_test = X_test.drop(labels=protected_labels, axis=1)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Ensemble(version=7, params={})
model.train(X_train, y_train)
all_predictions = model.test(X_test)
all_accuracies = [accuracy_score(y_test, all_predictions[:, i]) for i in range(len(all_predictions[0]))]
#each_predictions, agg_predictions = model.test_agg(X_test, disagg=True)
probabilities = model.test_proba(X_test)
if args.row in [3, 4]:
    pickle.dump([all_predictions, all_accuracies, probabilities], open('results/{0}_row{1}_{2}.pkl'.format(args.dataset, args.row, 'min' if args.minority else 'maj'), 'wb'))
else:
    pickle.dump([all_predictions, all_accuracies, probabilities], open('results/{0}_row{1}.pkl'.format(args.dataset, args.row), 'wb'))



