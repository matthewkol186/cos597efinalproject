from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd

# pre-processing and setting up the data
min_max_scaler = preprocessing.MinMaxScaler()

train_data, test_data = None, None
for version in ['train', 'test']:
    if version == 'train':
        file_name = './Data/adult_income/adult.data'
    elif version == 'test':
        file_name = './Data/adult_income/adult_data.test'
    
    data = pd.read_csv(file_name, sep=' ', header=None)
    data = data.replace({'\$': '', ',': ''}, regex=True)
    
    data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
    
    label = data['label']
    for i in range(len(label)):
        if version == 'train':
            if (label[i] == '<=50K'):
                label[i] = 0 
            elif (label[i]=='>50K'):
                label[i] = 1 
        elif version == 'test':
            if (label[i] == '<=50K.'):
                label[i] = 0 
            elif (label[i]=='>50K.'):
                label[i] = 1 
    data['label'] = label

    data.replace('?', np.nan, inplace=True)
    data = data.dropna()

    if version == 'train':
        train_data = data.copy(deep=True)

        data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label'] ]=data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label']].astype(str).astype(int)
        data_cat = data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]
        data_cat = pd.get_dummies(data_cat)
        data = data.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1).join(data_cat)
        min_max_scaler.fit(data)
    elif version == 'test':
        test_data = data.copy(deep=True)
data = pd.concat([train_data, test_data])

data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label'] ]=data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label']].astype(str).astype(int)
data_cat = data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]
data_cat = pd.get_dummies(data_cat)
#data = data.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1).join(data_cat)
data = pd.concat([data.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1), data_cat], axis=1)

np_scaled = min_max_scaler.transform(data)
data_norm= pd.DataFrame(np_scaled, columns = data.columns)
train_data = data_norm[:30162]
test_data = data_norm[30162:]
print(train_data.shape, test_data.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train, y_train = train_data.drop(['label'], axis=1), train_data['label']
X_test, y_test = test_data.drop(['label'], axis=1), test_data['label']
pickle.dump([X_train, y_train, X_test, y_test], open('Data/adult_income/processed_data.pkl', 'wb'))
