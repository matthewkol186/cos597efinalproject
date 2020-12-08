import pickle
import numpy as np
from ber_metrics import BEREstimator
import argparse

parser = argparse.ArgumentParser(description='diagnosing some errors')
parser.add_argument('--dataset', type=str, default='adult', help='adult or compas-arrest or compas-violent')
args = parser.parse_args()

print(args)

for row in ['row1', 'row2', 'row3_min', 'row3_maj', 'row4_min', 'row4_maj']:

    all_predictions, all_accuracies, probabilities = pickle.load(open('results/{0}_{1}.pkl'.format(args.dataset, row), 'rb'))
    if args.dataset == 'adult':
        X_train, y_train, X_test, y_test = pickle.load(open('Data/adult_income/processed_data.pkl', 'rb'))
    elif args.dataset == 'compas-arrest':
        X_train, y_train, X_test, y_test = pickle.load(open('Data/compas/processed_arrest.pkl', 'rb'))
    elif args.dataset == 'compas-violent':
        X_train, y_train, X_test, y_test = pickle.load(open('Data/compas/processed_violent.pkl', 'rb'))
    else:
        assert NotImplementedError


    if row == 'row3_min':
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['race_Black']==1], y_train[X_train['race_Black']==1]
            X_test, y_test = X_test[X_test['race_Black']==1], y_test[X_test['race_Black']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['race_is_african_american']==1], y_train[X_train['race_is_african_american']==1]
            X_test, y_test = X_test[X_test['race_is_african_american']==1], y_test[X_test['race_is_african_american']==1]
    elif row == 'row3_maj':
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['race_White']==1], y_train[X_train['race_White']==1]
            X_test, y_test = X_test[X_test['race_White']==1], y_test[X_test['race_White']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['race_is_causasian']==1], y_train[X_train['race_is_causasian']==1]
            X_test, y_test = X_test[X_test['race_is_causasian']==1], y_test[X_test['race_is_causasian']==1]
    elif row == 'row4_min':
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['sex_Female']==1], y_train[X_train['sex_Female']==1]
            X_test, y_test = X_test[X_test['sex_Female']==1], y_test[X_test['sex_Female']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['female']==1], y_train[X_train['female']==1]
            X_test, y_test = X_test[X_test['female']==1], y_test[X_test['female']==1]
    elif row == 'row4_maj':
        if args.dataset == 'adult':
            X_train, y_train = X_train[X_train['sex_Male']==1], y_train[X_train['sex_Male']==1]
            X_test, y_test = X_test[X_test['sex_Male']==1], y_test[X_test['sex_Male']==1]
        elif 'compas' in args.dataset:
            X_train, y_train = X_train[X_train['female']==0], y_train[X_train['female']==0]
            X_test, y_test = X_test[X_test['female']==0], y_test[X_test['female']==0]
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    adult_estimator = BEREstimator(X_test, y_test)

    probabilities = probabilities[1]

    agg_predictions = probabilities.mean(axis=1).round()
    plur_ber = adult_estimator.bootstrap_ensemble(probabilities, ensemble_version='plurality')
    #plur_ber = 1
    mi_ber = adult_estimator.bootstrap_ensemble(probabilities, ensemble_version='mi')

    
    acc = np.mean(np.equal(agg_predictions, y_test))
    #plur_ber = adult_estimator.plurality_ensemble_bound(probabilities)
    #mi_ber = adult_estimator.mi_ensemble_bound(probabilities, this_y=y_test, ensemble_predictions=agg_predictions)

    print("{0} has BER: plur = {1}, mi = {2}".format(row, plur_ber, mi_ber))
    #print(row)
    print("Classification Error: {}".format(1.-acc))
                    
