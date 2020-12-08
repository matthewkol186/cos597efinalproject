import pickle
import numpy as np
from ber_metrics import BEREstimator
import argparse
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="diagnosing some errors")
parser.add_argument(
    "--dataset",
    type=str,
    default="adult",
    help="adult or compas-arrest or compas-violent",
)
args = parser.parse_args()

print(args)

for row in ["row1", "row2", "row3_min", "row3_maj", "row4_min", "row4_maj"]:

    all_predictions, all_accuracies, _, _, probabilities = pickle.load(
        open("results/{0}_{1}.pkl".format(args.dataset, row), "rb")
    )
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = pickle.load(
            open("Data/adult_income/processed_data.pkl", "rb")
        )
    elif args.dataset == "compas-arrest":
        X_train, y_train, X_test, y_test = pickle.load(
            open("Data/compas/processed_arrest.pkl", "rb")
        )
    elif args.dataset == "compas-violent":
        X_train, y_train, X_test, y_test = pickle.load(
            open("Data/compas/processed_violent.pkl", "rb")
        )
    elif args.dataset == "mimic":
        X_train, X_test, y_train, y_test = pickle.load(
            open("mimic_compressed.pkl", "rb")
        )
    else:
        assert NotImplementedError

    if row == "row3_min":
        if args.dataset == "adult":
            X_train, y_train = (
                X_train[X_train["race_Black"] == 1],
                y_train[X_train["race_Black"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["race_Black"] == 1],
                y_test[X_test["race_Black"] == 1],
            )
        elif "compas" in args.dataset:
            X_train, y_train = (
                X_train[X_train["race_is_african_american"] == 1],
                y_train[X_train["race_is_african_american"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["race_is_african_american"] == 1],
                y_test[X_test["race_is_african_american"] == 1],
            )
        elif args.dataset == "mimic":
            X_train, y_train = (
                X_train[X_train["IS_RACE_BLACK"] == 1],
                y_train[X_train["IS_RACE_BLACK"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["IS_RACE_BLACK"] == 1],
                y_test[X_test["IS_RACE_BLACK"] == 1],
            )
        else:
            assert NotImplementedError
    elif row == "row3_maj":
        if args.dataset == "adult":
            X_train, y_train = (
                X_train[X_train["race_White"] == 1],
                y_train[X_train["race_White"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["race_White"] == 1],
                y_test[X_test["race_White"] == 1],
            )
        elif "compas" in args.dataset:
            X_train, y_train = (
                X_train[X_train["race_is_causasian"] == 1],
                y_train[X_train["race_is_causasian"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["race_is_causasian"] == 1],
                y_test[X_test["race_is_causasian"] == 1],
            )
        elif args.dataset == "mimic":
            X_train, y_train = (
                X_train[X_train["IS_RACE_WHITE"] == 1],
                y_train[X_train["IS_RACE_WHITE"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["IS_RACE_WHITE"] == 1],
                y_test[X_test["IS_RACE_WHITE"] == 1],
            )
        else:
            assert NotImplementedError
    elif row == "row4_min":
        if args.dataset == "adult":
            X_train, y_train = (
                X_train[X_train["sex_Female"] == 1],
                y_train[X_train["sex_Female"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["sex_Female"] == 1],
                y_test[X_test["sex_Female"] == 1],
            )
        elif "compas" in args.dataset:
            X_train, y_train = (
                X_train[X_train["female"] == 1],
                y_train[X_train["female"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["female"] == 1],
                y_test[X_test["female"] == 1],
            )
        elif args.dataset == "mimic":
            X_train, y_train = (
                X_train[X_train["IS_SEX_F"] == 1],
                y_train[X_train["IS_SEX_F"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["IS_SEX_F"] == 1],
                y_test[X_test["IS_SEX_F"] == 1],
            )
        else:
            assert NotImplementedError
    elif row == "row4_maj":
        if args.dataset == "adult":
            X_train, y_train = (
                X_train[X_train["sex_Male"] == 1],
                y_train[X_train["sex_Male"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["sex_Male"] == 1],
                y_test[X_test["sex_Male"] == 1],
            )
        elif "compas" in args.dataset:
            X_train, y_train = (
                X_train[X_train["female"] == 0],
                y_train[X_train["female"] == 0],
            )
            X_test, y_test = (
                X_test[X_test["female"] == 0],
                y_test[X_test["female"] == 0],
            )
        elif args.dataset == "mimic":
            X_train, y_train = (
                X_train[X_train["IS_SEX_M"] == 1],
                y_train[X_train["IS_SEX_M"] == 1],
            )
            X_test, y_test = (
                X_test[X_test["IS_SEX_M"] == 1],
                y_test[X_test["IS_SEX_M"] == 1],
            )
        else:
            assert NotImplementedError
    og_X_test, og_y_test = X_test.copy(), y_test.copy()
    print("IS_RACE_WHITE" in og_X_test.columns)
    X_train, y_train, X_test, y_test = (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
    )

    adult_estimator = BEREstimator(X_test, y_test)
    print(probabilities.shape)
    probabilities = probabilities[1]
    print(probabilities.shape)

    agg_predictions = probabilities.mean(axis=1).round()
    plur_ber = adult_estimator.bootstrap_ensemble(
        probabilities, ensemble_version="plurality"
    )
    mi_ber = adult_estimator.bootstrap_ensemble(probabilities, ensemble_version="mi")
    acc = np.mean(np.equal(agg_predictions, y_test))
    # plur_ber = adult_estimator.plurality_ensemble_bound(probabilities)
    # mi_ber = adult_estimator.mi_ensemble_bound(probabilities, this_y=y_test, ensemble_predictions=agg_predictions)
    probabilities = np.array(probabilities)
    print("{0} has BER: plur = {1}, mi = {2}".format(row, plur_ber, mi_ber))
    print("Classification Error: {}".format(1.0 - acc))

    tn, fp, fn, tp = confusion_matrix(y_test, agg_predictions).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print("FNR: {0}, FPR: {1}".format(fnr, fpr))
    if row in ["row1", "row2"]:
        portions = ["White", "Black", "Male", "Female"]
        for j in range(len(portions)):
            print(portions[j])
            if args.dataset == "adult":
                if j == 0:
                    the_key = "race_White"
                elif j == 1:
                    the_key = "race_Black"
                elif j == 2:
                    the_key = "sex_Male"
                elif j == 3:
                    the_key = "sex_Female"
            elif "compas" in args.dataset:
                if j == 0:
                    the_key = "race_is_causasian"
                elif j == 1:
                    the_key = "race_is_african_american"
                elif j == 2:
                    the_key = "female"
                elif j == 3:
                    the_key = "female"
            elif args.dataset == "mimic":
                if j == 0:
                    the_key = "IS_RACE_WHITE"
                elif j == 1:
                    the_key = "IS_RACE_BLACK"
                elif j == 2:
                    the_key = "IS_SEX_M"
                elif j == 3:
                    the_key = "IS_SEX_F"

            if "compas" in args.dataset and j == 2:
                this_probabilities, this_y_test = (
                    probabilities[np.where(np.array(og_X_test[the_key]) == 0)[0]],
                    y_test[og_X_test[the_key] == 0],
                )
            else:
                this_probabilities, this_y_test = (
                    probabilities[np.where(np.array(og_X_test[the_key]) == 1)[0]],
                    y_test[og_X_test[the_key] == 1],
                )
            agg_predictions = this_probabilities.mean(axis=1).round()
            adult_estimator = BEREstimator(X_test, y_test)
            plur_ber = adult_estimator.bootstrap_ensemble(
                this_probabilities, ensemble_version="plurality"
            )
            mi_ber = adult_estimator.bootstrap_ensemble(
                this_probabilities, ensemble_version="mi"
            )
            acc = np.mean(np.equal(agg_predictions, this_y_test))
            print("{0} has BER: plur = {1}, mi = {2}".format(row, plur_ber, mi_ber))
            print("Classification Error: {}".format(1.0 - acc))
            tn, fp, fn, tp = confusion_matrix(this_y_test, agg_predictions).ravel()
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            print("FNR: {0}, FPR: {1}".format(fnr, fpr))
    elif "min" in row:
        maj_row = row[:-2]
        maj_row = maj_row + "aj"
        _, _, _, _, maj_probabilities = pickle.load(
            open("results/{0}_{1}.pkl".format(args.dataset, maj_row), "rb")
        )
        agg_probabilities = np.concatenate(
            [probabilities, maj_probabilities[1]], axis=0
        )

        if args.dataset == "adult":
            X_train, y_train, X_test, y_test = pickle.load(
                open("Data/adult_income/processed_data.pkl", "rb")
            )
        elif args.dataset == "compas-arrest":
            X_train, y_train, X_test, y_test = pickle.load(
                open("Data/compas/processed_arrest.pkl", "rb")
            )
        elif args.dataset == "compas-violent":
            X_train, y_train, X_test, y_test = pickle.load(
                open("Data/compas/processed_violent.pkl", "rb")
            )
        elif args.dataset == "mimic":
            X_train, X_test, y_train, y_test = pickle.load(
                open("mimic_compressed.pkl", "rb")
            )
        else:
            assert NotImplementedError

        if "row3" in row:
            if args.dataset == "adult":
                y_test_min = y_test[X_test["race_Black"] == 1]
            elif "compas" in args.dataset:
                y_test_min = y_test[X_test["race_is_african_american"] == 1]
            elif args.dataset == "mimic":
                y_test_min = y_test[X_test["IS_RACE_BLACK"] == 1]
            if args.dataset == "adult":
                y_test = y_test[X_test["race_White"] == 1]
            elif "compas" in args.dataset:
                y_test = y_test[X_test["race_is_causasian"] == 1]
            elif args.dataset == "mimic":
                y_test = y_test[X_test["IS_RACE_WHITE"] == 1]
            y_test_min, y_test_maj = np.array(y_test_min), np.array(y_test)
            y_test_agg = np.concatenate([y_test_min, y_test_maj], axis=0)
        elif "row4" in row:
            if args.dataset == "adult":
                y_test_min = y_test[X_test["sex_Female"] == 1]
            elif "compas" in args.dataset:
                y_test_min = y_test[X_test["female"] == 1]
            elif args.dataset == "mimic":
                y_test_min = y_test[X_test["IS_SEX_F"] == 1]
            if args.dataset == "adult":
                y_test = y_test[X_test["sex_Male"] == 1]
            elif "compas" in args.dataset:
                y_test = y_test[X_test["female"] == 0]
            elif args.dataset == "mimic":
                y_test = y_test[X_test["IS_SEX_M"] == 1]
            y_test_min, y_test_maj = np.array(y_test_min), np.array(y_test)
            y_test_agg = np.concatenate([y_test_min, y_test_maj], axis=0)
        X_train, y_train, X_test, y_test = (
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test),
        )

        adult_estimator = BEREstimator(X_test, y_test_agg)
        agg_predictions = agg_probabilities.mean(axis=1).round()
        plur_ber = adult_estimator.bootstrap_ensemble(
            agg_probabilities, ensemble_version="plurality"
        )
        mi_ber = adult_estimator.bootstrap_ensemble(
            agg_probabilities, ensemble_version="mi"
        )
        acc = np.mean(np.equal(agg_predictions, y_test_agg))

        print("{0} has BER: plur = {1}, mi = {2}".format("combined", plur_ber, mi_ber))
        print("Classification Error: {}".format(1.0 - acc))
        tn, fp, fn, tp = confusion_matrix(y_test_agg, agg_predictions).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        print("FNR: {0}, FPR: {1}".format(fnr, fpr))

    print("------")
