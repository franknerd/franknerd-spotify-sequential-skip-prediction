import glob
import pandas as pd
from lightgbm import LGBMClassifier
import time
from tqdm import tqdm
import re
import pathlib
import gc


start_time = time.time()

''' file path '''
train_path = glob.glob('train_for_model_knn/*')
test_path = glob.glob('test_for_model_knn/*')

if len(train_path)>0 and len(test_path):
    train_path.sort()
    test_path.sort()

    for path in train_path:
         print(path)

    for path in test_path:
         print(path)

    print('the number of train files:',len(train_path))
    print('the number of test files:', len(test_path ))

else:
    print('cannot find any file!')
    quit()

date_of_name = []
for i in range(len(train_path)):
    name = re.findall('2018\d{4}',train_path[i])
    date_of_name.append(name[0])

# ''' select the number of files '''
# print('')
# print('- - - - - type the number of files you want to do modelling - - - - - ')
# print('- - - - - type \'all\' means choosing all files- - - - - ')
#
# number = input()
# if number == 'all' or 'ALL':
#     num_file = int(len(number))
# else:
#     num_file = number
# #print('num_file',num_file)

start = 1
#end = 1
end = len(train_path)

train_fields = ['skip_2','duration','release_year',
'us_popularity_estimate','acousticness','beat_strength','bounciness','danceability','dyn_range_mean','energy',
'flatness','instrumentalness','key','liveness','loudness','mechanism','organism','speechiness','tempo',
'time_signature','valence','acoustic_vector_0','acoustic_vector_1','acoustic_vector_2','acoustic_vector_3',
'acoustic_vector_4','acoustic_vector_5','acoustic_vector_6','acoustic_vector_7','skip_2_avg_knn']

train_dtypes = {
'skip_2'            :        'int8',
'duration'          :        'float16',
'release_year'      :        'int8',
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
'skip_2_avg_knn'        :        'float16'}


test_fields = ['session_position','session_length','duration','release_year',
'us_popularity_estimate','acousticness','beat_strength','bounciness','danceability','dyn_range_mean','energy',
'flatness','instrumentalness','key','liveness','loudness','mechanism','organism','speechiness','tempo',
'time_signature','valence','acoustic_vector_0','acoustic_vector_1','acoustic_vector_2','acoustic_vector_3',
'acoustic_vector_4','acoustic_vector_5','acoustic_vector_6','acoustic_vector_7','skip_2_avg_knn']


''' make directories '''
# make a directory if there are no directories
pathlib.Path('output_1224_knn_1').mkdir(parents=True, exist_ok=True)

''' for each day build a model '''
for i in tqdm(range(0,20,1)):

    output = ''

    ''' import dataframe '''
    train = pd.read_csv(train_path[i],dtype=train_dtypes, usecols = train_fields)
    X_train = train.loc[:, train.columns != 'skip_2']
    y_train = train['skip_2']

    X_test = pd.read_csv(test_path[i],usecols = test_fields)

    clf = LGBMClassifier(n_estimators=100,
                        objective='binary',
                        learning_rate = 0.05,
                        n_jobs=-1,
                        random_state=42)
    X_test_in_glbm = X_test[['duration','release_year',
    'us_popularity_estimate','acousticness','beat_strength','bounciness','danceability','dyn_range_mean','energy',
    'flatness','instrumentalness','key','liveness','loudness','mechanism','organism','speechiness','tempo',
    'time_signature','valence','acoustic_vector_0','acoustic_vector_1','acoustic_vector_2','acoustic_vector_3',
    'acoustic_vector_4','acoustic_vector_5','acoustic_vector_6','acoustic_vector_7','skip_2_avg_knn']]

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test_in_glbm)

    ''' file format '''
    position = X_test['session_position'].tolist()
    length   = X_test['session_length'].tolist()

    for pos in range(len(y_pred)) :
        output += str(y_pred[pos])
        if position[pos] == length[pos]:
            output += '\n'

    '''save file'''
    output_path = 'output_1224_knn_1/'+date_of_name[i]+'.txt'
    output_file = open(output_path, 'w')
    output_file.write(output)
    output_file.close()

    del y_train; del train; del X_train; del X_test;
    del position; del length;
    gc.collect()

print("--- %s mins ---" % ((time.time() - start_time)/60))

'''merge all output into one'''
# out_path = glob.glob('output_1224_knn_1/*')
# out_path.sort()
#
# total =''
# for each_file in out_path:
#     with open(each_file, 'r') as file:
#         data =file.read()
#         total += data
#
# output_file = open('output_1224_knn_1/submission.txt', 'w')
# output_file.write(total)
# output_file.close()

print("--- %s mins ---" % ((time.time() - start_time)/60))
