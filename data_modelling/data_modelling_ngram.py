import glob
import pandas as pd
import time
from tqdm import tqdm
import re
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn import preprocessing

from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

'''import '''
track_feature = pd.read_csv('track_merge/track_all.csv')
track_id = track_feature['track_id'].tolist()

'''one hot '''
track_feature = pd.get_dummies(track_feature, prefix=['mode'], columns=['mode'])
track_feature['mode_major'] = track_feature['mode_major'].astype('int8')
track_feature.drop(columns=['mode_minor'],inplace= True)

'''normalisation'''
scaler =  preprocessing.MinMaxScaler()
df_track_scaled = pd.DataFrame(scaler.fit_transform(track_feature.iloc[:,1:]), columns=track_feature.columns[1:])

kmeans = KMeans(n_clusters=10, random_state=42,n_jobs=-1).fit(df_track_scaled)
lable = kmeans.predict(df_track_scaled)
df_track_scaled['group'] = lable
df_track_scaled['track_id'] = track_id
df_track_scaled =  df_track_scaled[['track_id','group']]

''' make directories '''
pathlib.Path('output_0104_abc').mkdir(parents=True, exist_ok=True)

''' file path'''
path = glob.glob('test_set/*')

if len(path)>0:
    train_path = [ i for i in path if 'prehistory' in i ]
    test_path = list(set(path)-set(train_path))
    train_path.sort()
    test_path.sort()
    print('the number of train files:',len(train_path))
    print('the number of test files:', len(test_path ))
else:
    print('cannot find any file!')
    quit()


date_of_name = []
for i in range(len(train_path)):
    name = re.findall('2018\d{4}',train_path[i])
    date_of_name.append(name[0])

train_field = ['track_id_clean','session_position','session_length','skip_2']
test_field = ['track_id_clean','session_position','session_length']

for i_file in tqdm(range(0,30,1)):

    df_train  = pd.read_csv(train_path[i_file], usecols = train_field)
    df_test  = pd.read_csv(test_path[i_file], usecols = test_field)
    df_train.rename(columns={'track_id_clean':'track_id'}, inplace=True)
    df_test.rename(columns= {'track_id_clean':'track_id'}, inplace=True)

    '''merge'''
    df_train = pd.merge(df_train, df_track_scaled, on=['track_id'],how='left')
    df_test  = pd.merge(df_test, df_track_scaled, on=['track_id'],how='left')

    '''append'''
    df_train['groupb'] =np.concatenate((np.array(df_train['group'])[-1], np.array(df_train['group'])[0:-1]), axis=None)
    df_train['groupc'] =np.concatenate((np.array(df_train['groupb'])[-1], np.array(df_train['groupb'])[0:-1]), axis=None)

    df_test['groupb'] =np.concatenate((np.array(df_test['group'])[-1], np.array(df_test['group'])[0:-1]), axis=None)
    df_test['groupc'] =np.concatenate((np.array(df_test['groupb'])[-1], np.array(df_test['groupb'])[0:-1]), axis=None)

    df_train = df_train[df_train['session_position']>2]

    # drop
    df_train.drop(['track_id'],axis=1,inplace=True)
    df_test.drop(['track_id'],axis=1,inplace=True)

    #category
    df_train['group']  = df_train['group'].astype('category')
    df_train['groupb'] = df_train['groupb'].astype('category')
    df_train['groupc'] = df_train['groupc'].astype('category')

    df_test['group']  = df_test['group'].astype('category')
    df_test['groupb'] = df_test['groupb'].astype('category')
    df_test['groupc'] = df_test['groupc'].astype('category')

    # int
    df_train['skip_2']  = df_train['skip_2'].astype('int')


    X_train = df_train[['group','groupb','groupc']].values
    y_train = df_train['skip_2']
    X_test = df_test[['group','groupb','groupc']].values

    # model
    clf = LGBMClassifier(n_estimators=100,
                    objective='binary',
                    learning_rate = 0.05,
                    n_jobs=-1,
                    random_state=42)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    position = df_test['session_position'].tolist()
    length   = df_test['session_length'].tolist()

    output = ''
    for pos in range(len(y_pred)) :
        output += str(y_pred[pos])
        if position[pos] == length[pos]:
            output += '\n'

    '''save file'''
    output_path = 'output_0104_abc/'+date_of_name[i_file]+'.txt'
    output_file = open(output_path, 'w')
    output_file.write(output)
    output_file.close()

    del df_train; del df_test;
    del X_train; del X_test;
    del length; del position
    gc.collect()
