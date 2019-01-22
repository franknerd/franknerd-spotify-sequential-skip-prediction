import glob
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import statistics
import pathlib


from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.models import Sequential
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

''' make directories '''
# make a directory if there are no directories
pathlib.Path('output_0103_lstm').mkdir(parents=True, exist_ok=True)

''' glob path'''
train_path = glob.glob('training_lstm/*')
train_path.sort()

date_of_name = []
for i in range(len(train_path)):
    name = re.findall('2018\d{4}',train_path[i])
    date_of_name.append(name[0])
date_of_name.sort()

train_dtypes = {
'session_position'  :        'int8',
'session_length'    :        'int8',
'skip_2'            :        'int8',
'duration'          :        'float16',
'us_popularity_estimate':    'float16',
'acousticness'      :        'float16',
'beat_strength'     :        'float16',
'bounciness'        :        'float16',
'danceability'      :        'float16',
'dyn_range_mean'    :        'float16',
'energy'            :        'float16',
'flatness'          :        'float16',
'instrumentalness'  :        'float16',
'key'               :        'int8',
'liveness'          :        'float16',
'loudness'          :        'float16',
'mechanism'         :        'float16',
'mode'              :        'object',
'organism'          :        'float16',
'speechiness'       :        'float16',
'tempo'             :        'float16',
'time_signature'    :        'int8',
'valence'           :        'float16',
'acoustic_vector_0' :        'float16',
'acoustic_vector_1' :        'float16',
'acoustic_vector_2' :        'float16',
'acoustic_vector_3' :        'float16',
'acoustic_vector_4' :        'float16',
'acoustic_vector_5' :        'float16',
'acoustic_vector_6' :        'float16',
'acoustic_vector_7' :        'float16',
'skip_1_avg'        :        'float16',
'skip_2_avg'        :        'float16',
'skip_3_avg'        :        'float16',
}

test_dtypes ={
'session_position'  :        'int8',
'session_length'    :        'int8',
'duration'          :        'float16',
'us_popularity_estimate':    'float16',
'acousticness'      :        'float16',
'beat_strength'     :        'float16',
'bounciness'        :        'float16',
'danceability'      :        'float16',
'dyn_range_mean'    :        'float16',
'energy'            :        'float16',
'flatness'          :        'float16',
'instrumentalness'  :        'float16',
'key'               :        'int8',
'liveness'          :        'float16',
'loudness'          :        'float16',
'mechanism'         :        'float16',
'mode'              :        'object',
'organism'          :        'float16',
'speechiness'       :        'float16',
'tempo'             :        'float16',
'time_signature'    :        'int8',
'valence'           :        'float16',
'acoustic_vector_0' :        'float16',
'acoustic_vector_1' :        'float16',
'acoustic_vector_2' :        'float16',
'acoustic_vector_3' :        'float16',
'acoustic_vector_4' :        'float16',
'acoustic_vector_5' :        'float16',
'acoustic_vector_6' :        'float16',
'acoustic_vector_7' :        'float16',
'skip_1_avg'        :        'float16',
'skip_2_avg'        :        'float16',
'skip_3_avg'        :        'float16',
}

train_fields = ['session_position','session_length','skip_2','duration',
                'us_popularity_estimate','acousticness','beat_strength','bounciness',
                'danceability','dyn_range_mean','energy',
                'flatness','instrumentalness','key','liveness','loudness',
                'mechanism','organism','speechiness','tempo',
                'time_signature','valence','acoustic_vector_0','acoustic_vector_1',
                'acoustic_vector_2','acoustic_vector_3',
                'acoustic_vector_4','acoustic_vector_5',
                'acoustic_vector_6','acoustic_vector_7','skip_1_avg','skip_2_avg','skip_3_avg']


test_fields = ['session_position','session_length',
               'duration', 'us_popularity_estimate', 'acousticness',
               'beat_strength', 'bounciness', 'danceability', 'dyn_range_mean',
               'energy', 'flatness', 'instrumentalness', 'key', 'liveness', 'loudness',
               'mechanism', 'organism', 'speechiness', 'tempo', 'time_signature',
               'valence', 'acoustic_vector_0', 'acoustic_vector_1',
               'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
               'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7',
               'skip_1_avg','skip_2_avg','skip_3_avg']

'''test file'''
first_half_path  = glob.glob('train_for_model/*')
second_half_path = glob.glob('test_for_model/*')
first_half_path.sort()
second_half_path.sort()

session_path   = glob.glob('test_set/*')
first_half_session_path = [ i for i in session_path if 'prehistory' in i ]
second_half_session_path = list(set(session_path)-set(first_half_session_path))
first_half_session_path.sort()
second_half_session_path.sort()

for file_i in tqdm(range(0,66,1)):

    ''' training set'''
    df = pd.read_csv(train_path[file_i],dtype=train_dtypes,usecols=train_fields)

    ''' normalization'''
    X = df[list(df.columns[[0,1]])+list(df.columns[3:])].values
    y =  np.array(df['skip_2'])

    for i in range(X.shape[1]):
        data = X[:,i].reshape(-1, 1)
        minmax_scale = MinMaxScaler().fit(data)
        df_minmax = minmax_scale.transform(data)
        data_reshape = df_minmax.ravel()
        X[:,i] = data_reshape


    '''reshape'''
    position = np.array(df['session_position'])
    length = np.array(df['session_length'])

    total_x = []
    start = 0
    for i in range(len(position)):
        if position[i] == length[i]:
            end = i
            total_x.append(X[start:end+1,:])
            start = i+1

    total_y = []
    start = 0
    for i in range(len(position)):
        if position[i] == length[i]:
            end = i
            total_y.append(y[start:end+1])
            start = i+1

    total_x = np.array(total_x)
    total_y = np.array(total_y)

    X_train = pad_sequences(total_x, padding='post',dtype = "float32")
    y_train = pad_sequences(total_y, padding='post')

    y_train = y_train.reshape(y_train.shape[0],20,1)

    '''test set'''

    session_field = ['session_id']
    first_half =  pd.read_csv(first_half_path[file_i],usecols=test_fields,dtype = test_dtypes)
    second_half = pd.read_csv(second_half_path[file_i],usecols=test_fields,dtype = test_dtypes)
    first_half_session  = pd.read_csv(first_half_session_path[file_i],usecols=session_field)
    second_half_session = pd.read_csv(second_half_session_path[file_i],usecols=session_field)
    first_half['session_id'] = first_half_session
    second_half['session_id']= second_half_session

    test = pd.concat([first_half,second_half])
    test.sort_values(by=['session_id','session_position'],inplace=True)
    test.reset_index(drop=True,inplace=True)
    test.drop(['session_id'],axis=1,inplace=True)

    '''test pos and length'''
    test_pos = test['session_position'].tolist()
    test_length = test['session_length'].tolist()

    sec_half_position = np.array(second_half['session_position'])
    sec_half_length = np.array(second_half['session_length'])

    test = np.array(test)

    '''normalization'''
    for i in range(test.shape[1]):
        data = test[:,i].reshape(-1, 1)
        minmax_scale = MinMaxScaler().fit(data)
        df_minmax = minmax_scale.transform(data)
        data_reshape = df_minmax.ravel()
        test[:,i] = data_reshape

    '''reshape'''
    total_test = []
    start = 0
    for i in range(len(test_pos)):
        if test_pos[i] == test_length[i]:
            end = i
            total_test.append(test[start:end+1,:])
            start = i+1

    total_test = np.array(total_test)
    X_test = pad_sequences(total_test, padding='post',dtype = "float32")

    '''LSTM'''
    model = Sequential()
    model.add(LSTM(100, input_shape=(20,32),return_sequences=True))
    model.add(LSTM(1,return_sequences=True))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=10000, epochs=5,verbose=2)
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[1])

    y_temp = y_pred.reshape(-1)
    median_value = statistics.median(y_temp)

    '''finally'''
    position_list = []
    sub_list = []
    for i in range(len(sec_half_position)):
        if sec_half_position[i] < sec_half_length[i]:
            sub_list.append(sec_half_position[i])
        else:
            sub_list.append(sec_half_position[i])
            position_list.append(sub_list)
            sub_list = []

    output = ''
    for line_num in range(len(y_pred)):
        length = len(position_list[line_num])
        for i in position_list[line_num]:
            pos = i-1
            if y_pred[line_num][pos]>= median_value:
                output += str(1)
            else:
                output += str(0)
        output += '\n'

    output_path = 'output_0103_lstm/'+date_of_name[file_i]+'.txt'
    output_file = open(output_path, 'w')
    output_file.write(output)
    output_file.close()
