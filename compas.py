#from responsibly.dataset import COMPASDataset
#
#compas_ds = COMPASDataset()
#df = compas_ds.df

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
# CSV's from https://github.com/charliemarx/pmtools/blob/master/data/compas_arrest_processed.csv

arrest = pd.read_csv('Data/compas/compas_arrest_processed.csv')
violent = pd.read_csv('Data/compas/compas_violent_processed.csv')

X_arrest, y_arrest = arrest.drop(['arrest'], axis=1), arrest['arrest']
X_violent, y_violent = violent.drop(['violent'], axis=1), violent['violent']

X_train, X_test, y_train, y_test = train_test_split(X_arrest, y_arrest, test_size=0.33)
pickle.dump([X_train, y_train, X_test, y_test], open('Data/compas/processed_arrest.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X_violent, y_violent, test_size=0.33)
pickle.dump([X_train, y_train, X_test, y_test], open('Data/compas/processed_violent.pkl', 'wb'))
