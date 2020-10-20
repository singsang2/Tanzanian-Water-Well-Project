import pandas as pd
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
import pickle

def save_data(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_percent_retained(df):
    original_length = 59400
    print(f'Original Length: {original_length}')
    print(f'Current Length: {df.shape[0]}')
    print(f'Percent Retained: {round(df.shape[0]/original_length * 100, 2)}%')

def drop_rows_na(df, col):
    indices = df[col].dropna().index
    return df.loc[indices,:]

def evaluate_clf_model(model, X_train, y_train, X_test,y_test, features=None, classes=None, prob=True,
                           normalize='true',cmap='Purples', label='', cv=0):
    """Accepts an sklearn-compatible classification model + test data 
    and displays several sklearn.metrics functions: 
    - classifciation_report
    - plot_confusion_matrix
    - plot_roc_curve
    """
     
    ## Get Predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    if prob:
        y_proba_train = model.predict_proba(X_train)
        auc_train = metrics.roc_auc_score(y_train, y_proba_train, average='weighted', multi_class='ovr')
        y_proba_test = model.predict_proba(X_test)
        auc_test = metrics.roc_auc_score(y_test, y_proba_test, average='weighted', multi_class='ovr')
    
    if cv!=0:
        cv_score = cross_val_score(model, pd.concat([X_train, X_test]), 
                                   np.concatenate((y_train, y_test), axis=None), 
                                   cv=cv)
    ## Classification Report / Scores 
    table_header = "[i] CLASSIFICATION REPORT"
    
    ## Add Label if given
    if len(label)>0:
        table_header += f" {label}"
    ## PRINT CLASSIFICATION REPORT
    dashes = '---'*20
    print(dashes,table_header,dashes,sep='\n')    
    print('Train Accuracy : ', round(metrics.accuracy_score(y_train, y_hat_train),4))
    print('Test Accuracy : ', round(metrics.accuracy_score(y_test, y_hat_test),4))
    if prob:
        print('Train AUC : ', round(auc_train,4))
        print('Test AUC : ', round(auc_test,4))
    if cv:
        print(f'CV score (n={cv})', round(np.mean(cv_score), 4))
    print(dashes+'\n')
    
    

    print(metrics.classification_report(y_test,y_hat_test,
                                    target_names=classes))
    
    report = metrics.classification_report(y_test,y_hat_test,
                                               target_names=classes,
                                          output_dict=True)
    print(dashes+"\n\n")
    
    

    ## MAKE FIGURE
    fig, ax = plt.subplots(figsize=(10,4))
    ax.grid(False)
    
    ## Plot Confusion Matrix 
    metrics.plot_confusion_matrix(model, X_test,y_test,
                                  display_labels=classes,
                                  normalize=normalize,
                                  cmap=cmap,ax=ax)
    ax.set(title='Confusion Matrix')
    
    if label != 'KNN':
        # Feature Importance
        fig, ax = plt.subplots(figsize=(10,4))

        # get features if not given
        if features==None:
            features = X_train.keys()

        feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)[:10]
        feat_imp.plot(kind='barh', title='Feature Importances')
        ax.set(ylabel='Feature Importance Score')
        ax.invert_yaxis() # this will make the graphs in order

    plt.show()
    
    return report, fig, ax

def import_data(file1, file2, drop_cols=None):
    X = pd.read_csv(file1)
    y = pd.read_csv(file2)
    
    X['date_recorded'] = pd.to_datetime(X['date_recorded'])
    X['year_recorded'] = X['date_recorded'].apply(lambda x: x.year)
    X['month_recorded'] = X['date_recorded'].apply(lambda x: x.month)
    X.drop(columns=['date_recorded'], axis=1, inplace=True)

    # Changes 0 to recorded year for construction year
    mask = X['construction_year'] == 0
    X['construction_year'][mask] = X['year_recorded'][mask]

    # 1 if a name exists and 0 if not.
    X['wpt_name'] = X['wpt_name']!='none'
    
    if drop_cols != None:
        X.drop(columns=drop_cols, axis=1, inplace=True)
    
    df = pd.concat([X, y[['status_group']]], axis=1)
    
    cols = ['permit', 'public_meeting']
    for col in cols:
        df = drop_rows_na(df, col)
        df = df.reset_index().drop(columns='index')
    
    # get rid of rows with 0 lat and 0 long
    df.drop(df[(df['latitude']>-0.5)].index, inplace=True)
    df = df.reset_index().drop(columns='index')
    
    get_percent_retained(df)
    return df

def get_distance(x, y):
    """
    x = [lat and long]
    y = array of lat and long
    """
    pi = np.pi
    R = 6371 # Average Earth radius in km
    
    # converts to radians
    lat1 = x[0]*pi/180 
    lat2 = y.iloc[:,0]*pi/180
    long1 = x[1]*pi/180
    long2 = y.iloc[:,1]*pi/180

    # finds delta angles
    del_lat = lat1 - lat2
    del_long = long1 - long2
        
    # Source: https://www.movable-type.co.uk/
    a = np.sin(del_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(del_long/2)**2
    c = 2*np.arctan(np.sqrt(a)/np.sqrt(1-a))
    d= R*c # distance in km
        
    return d

def get_nearby(df, r=30):
    X = df.drop(columns='status_group', axis=1).copy()
    y = df[['status_group']].copy()

    # Calculates percentage of functional, nonfunctional, and need repair wells within certain radius (30 km)
    perc_func_near = []
    perc_non_func_near = []
    perc_repair_near = []

    for _, row in X.iterrows():
    #     print(index, row['latitude'], row['longitude'])
        wp = [row['latitude'], row['longitude']]
        wps = X[['latitude', 'longitude']].copy()
        # print(wps, type(wps))

        d = get_distance(wp, wps)
        
        distance = r
        indices = X[d < distance].index
        

        perc_func = round(np.sum(y.iloc[indices, :] == 'functional')/len(indices)*100, 2)
        perc_non_func = round(np.sum(y.iloc[indices, :] == 'non functional')/len(indices)*100, 2)
        perc_repair_func = round(np.sum(y.iloc[indices, :] == 'functional needs repair')/len(indices)*100, 2)
        
        perc_func_near.append(perc_func[0])
        perc_non_func_near.append(perc_non_func[0])
        perc_repair_near.append(perc_repair_func[0])
    
    # Assigns percentages in appropriate columns
    df['perc_func_nearby'] = perc_func_near
    df['perc_non_func_nearby'] = perc_non_func_near
    df['perc_repair_nearby'] = perc_repair_near

    return df

def extraction(df):
    # Extraction dictionary to simplify some of trivial ones as 'other'
    extraction_dict = {'nira/tanira':'nira', 'swn 80':'other', 
                    'other - rope pump': 'other', 'other - swn 81': 'other',
                    'other - play pump': 'other', 'other - mkulima/shinyanga':'other',
                    'windmill': 'other', 'cemo':'other', 'climax':'other',
                    'walimi':'other', 'india mark iii':'other'}

    # replaces insignificant extractor types as 'other'
    df['extraction_type'] = df['extraction_type'].apply(lambda x: extraction_dict[x] if x in extraction_dict.keys() else x)
    return df

# makes a dictionary that has percent func/non func/repair for each installer
def get_percent(data, col='installer', drop=True):
    if col in data.keys():
        # copies the column
        sample = data[[col, 'status_group']].copy()

        # Get dummies for the column
        sample_dummies = pd.get_dummies(sample, columns=['status_group'])

        # Sum total functioning, nonfunctioning, and needing repair wells for each group
        a = sample_dummies.groupby(col).sum()

        a['total'] = a.sum(axis=1)

        # Get percentages
        a['perc_func'] = a['status_group_functional']/a['total']
        a['perc_repair'] = a['status_group_functional needs repair']/a['total']
        a['perc_non_func'] = a['status_group_non functional']/a['total']

        a = a[['perc_func', 'perc_repair', 'perc_non_func']].reset_index()

        perc_dict = {}
        for index, row in a.iterrows():
            perc_dict[row[col]] = [row['perc_func'], row['perc_repair'], row['perc_non_func']]

        data[f'{col}_perc_func'] = data[col].apply(lambda x: perc_dict[x][0] if x in perc_dict.keys() else np.NaN)
        data[f'{col}_perc_repair'] = data[col].apply(lambda x: perc_dict[x][1] if x in perc_dict.keys() else np.NaN)
        data[f'{col}_perc_non_func'] = data[col].apply(lambda x: perc_dict[x][2] if x in perc_dict.keys() else np.NaN)
        
        if drop:
            data.drop(columns=[col], axis=1, inplace=True)
        else:
            print(f'Column name {col} not found.')
    return perc_dict

def preprocess(df):
    # gets cat_cols and num_cols
    cat_cols = list(df.drop(columns='status_group', axis=1).select_dtypes('O').columns)
    cat_cols.append('wpt_name')
    num_cols = list(df.select_dtypes(['int64', 'float64']).columns)

    X = df.drop(columns='status_group', axis=1)
    y = df['status_group']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    # takes care of y
    encoder = LabelEncoder()
    y_train_tf = encoder.fit_transform(y_train)
    y_test_tf = encoder.transform(y_test)

    # takes care of X columns
    pipe_cat = Pipeline(steps=[('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),
                               ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

    pipe_num = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                               ('scale', MinMaxScaler())])

    preprocessing = ColumnTransformer(transformers=[('num', pipe_num, num_cols),
                                                 ('cat', pipe_cat, cat_cols)])
    
    X_train_tf = preprocessing.fit_transform(X_train)
    X_test_tf = preprocessing.transform(X_test)

    # gets feature names
    feature_names = preprocessing.named_transformers_['cat'].named_steps['encoder'].get_feature_names(cat_cols).tolist()
    # combines with numerical column names
    features = num_cols + feature_names

    X_train_tf = pd.DataFrame(X_train_tf, columns = features)
    X_test_tf = pd.DataFrame(X_test_tf, columns = features)

    return X_train_tf, X_test_tf, y_train_tf, y_test_tf

def do_everything(file1, file2, drop_cols, save=True):
    if save:
        df = import_data(file1, file2, drop_cols)
        df = get_nearby(df, r=30)
        df = extraction(df)

        # Gets percetnages for different installers
        installer_dict = get_percent(df, col='installer')
        # Gets percetnages for different funders
        funder_dict = get_percent(df, col='funder')

        save_data(installer_dict, 'datasets/installer_dict')
        save_data(funder_dict, 'datasets/funder_dict')
        
        drop_features = ['funder_perc_non_func', 'installer_perc_non_func', 'perc_non_func_nearby']
        df.drop(columns=drop_features, axis=1, inplace = True)

        save_data(df, 'datasets/final_df')

        X_train, X_test, y_train, y_test = preprocess(df)
        class_weight, class_weight_dict = get_class_weights(y_train)

        create_models(X_train, X_test, y_train, y_test, class_weight = class_weight_dict)
    else:
        installer_dict = load_data('datasets/installer_dict')
        funder_dict = load_data('datasets/funder_dict')
        df = load_data('datasets/final_df')
        final_models = load_data('datasets/final_models')
        
    print('done')


    return df, X_train, X_test, y_train, y_test

def get_class_weights(y):
    class_weight = list(compute_class_weight(class_weight='balanced', classes=[0, 1, 2], y=y))
    class_weight_dict = {0: class_weight[0],
                    1: class_weight[1],
                    2: class_weight[2]}
    return class_weight, class_weight_dict

def create_models(X_train, X_test, y_train, y_test, class_weight):
    # Random Forest
    RF = RandomForestClassifier(criterion='gini',
                            n_estimators=1000,
                            class_weight=class_weight,
                            max_depth = 15,
                            min_samples_split=0.001,
                            max_features='sqrt')
    RF.fit(X_train, y_train)

    # Linear Regressor
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(random_state = 0, 
                            max_iter = 1000)
    LR.fit(X_train, y_train)

    # XGBoost
    # from sklearn.model_selection import GridSearchCV
    XGB = XGBClassifier(objective='mutli:softmax', num_class=3, 
                        class_weight=class_weight,
                        n_estimators=200,
                        max_depth=5,
                        subsample=0.6,
                        min_child_weight=0.8)
                        
    XGB.fit(X_train, y_train)

    # Adaboost
    AB = AdaBoostClassifier(n_estimators=300)
    AB.fit(X_train, y_train)

    # Extra Tree
    ET = ExtraTreesClassifier(n_estimators=200,
                            class_weight=class_weight,
                            max_features=0.7,
                            min_samples_split=0.01)

    ET.fit(X_train, y_train)

    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    GB = GradientBoostingClassifier(n_estimators=300,
                                    max_depth=7,
                                    subsample=0.6)
    GB.fit(X_train, y_train)

    # KNN
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)


    save_data([RF, LR, XGB, AB, ET, GB, KNN], 'datasets/final_models')
    save_data(['RF', 'LR', 'XGB', 'AB', 'ET', 'GB', 'KNN'], 'datasets/final_model_names')

def create_graph(df):
    pass
