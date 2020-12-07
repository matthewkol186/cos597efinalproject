# code from Irene Chen

def get_salary_data():
    fieldnames =  ['age', 'workclass', 'fnlwgt', 'education', 
               'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'salary']

    # handicap_cols = ['sex', 'relationship', 'marital-status', 'race', 'occupation']
    # handicap_cols = list()

    ignore_cols = ['fnlwgt', 'salary', 'sex']
    df = pd.read_csv('./adult.data', header=None, names= fieldnames)

    feature_cols = [i for i in fieldnames if i not in ignore_cols]
    df2 = df[feature_cols]
    
    # hand engineer some columns
    df2.loc[~df['workclass'].isin([' Federal-gov', ' Local-gov', ' Private', ' Self-emp-inc']), 'workclass'] = 'No-inc'
    df2.loc[df['workclass'].isin([' Federal-gov', ' Local-gov']), 'workclass'] = 'Gov'
    df2.loc[df['workclass'].isin([' Private', ' Self-emp-inc']), 'workclass'] = 'Private'
    df2 = df2.drop(columns=['education']) # keep education-num
    df2.loc[df['native-country'].isin([' Cambodia', ' China', ' Hong', ' India', ' Japan', ' Philippines', ' Taiwan', ' Thailand', ' Vietnam', ' Laos', ' Iran']), 'native-country'] = 'Asia'
    df2.loc[df['native-country'].isin([' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' Guatemala', ' Honduras', ' Nicaragua', ' Haiti', ' Jamaica', ' Trinadad&Tobago', ' Peru', ' Mexico']), 'native-country'] = 'South America'
    df2.loc[df['native-country'].isin([' Canada', ' Greece', ' England', ' France', ' Germany', ' Italy', ' Holand-Netherlands', ' Ireland', ' Scotland', ' Portugal', ' Poland', ' Hungary', ' Yugoslavia']), 'native-country'] = 'Europe&Canada'
    df2.loc[df['native-country'].isin([' Puerto-Rico', ' Outlying-US(Guam-USVI-etc)']), 'native-country'] = 'US_non_state'
    df2.loc[df['race'].isin([' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Other']), 'race'] = "Other"

    df3 = pd.get_dummies(df2)

    # pdb.set_trace()
    df3['sex'] = df['sex'] == ' Female'
    df3['sex'] = df3['sex'].apply(int)

    df3['salary'] = df['salary'] == ' >50K'
    df3['salary'] = df3['salary'].apply(int)

    """ LR coefficients
    capital-gain      0.000228
    capital-loss      0.000868
    hours-per-week    0.011817
    age               0.022415
    native-country    0.106419
    education-num     0.123151
    workclass         0.318127
    education         0.392940
    occupation        0.413003
    race              0.469515
    marital-status    0.487364
    relationship      0.686120
    sex               0.997766
    """

    feat_cols = [i for i in df3.columns if i != 'salary' and i != 'sex']
    return df3, feat_cols, 'salary', 'sex'

df, feat_cols, a, b = get_salary_data()

def b_noise(data, target_col, feature_cols):
    # now noise estimation!
    d0 = data[data[target_col] == 0][feature_cols]
    d1 = data[data[target_col] == 1][feature_cols]

    mu0 = d0.mean(axis=0)
    mu1 = d1.mean(axis=0)

    cov0 = d0.cov().values
    cov1 = d1.cov().values

    p1 = np.expand_dims(mu1 - mu0,axis=1)
    p2 = np.linalg.inv((cov0 + cov1)/2)
    p3 = np.linalg.det((cov0 + cov1)/2)
    p4 = np.sqrt(np.linalg.det(cov0)*np.linalg.det(cov1))

    b_dist = 1./8 * np.transpose(p1).dot(p2).dot(p1) + 0.5 * np.log(p3 / p4)
    b_dist = b_dist[0][0]

    p_c0 = (data[target_col] == 0).mean()
    p_c1 = (data[target_col] == 1).mean()

    bound_low = 0.5 * (1 - np.sqrt(1 - 4 * p_c0 * p_c1 * np.exp(-2 * b_dist)))
    bound_up = np.exp(-b_dist) * np.sqrt(p_c0 * p_c1)

    print(bound_low)
    print(bound_up)
    
b_noise(df, 'salary', feat_cols)