import pandas as pd
import glob
import re
from tqdm import tqdm
import time
import gc



start_time = time.time()

train_path = glob.glob('training_set/*')
train_path.sort()

day = set()
for i in range(len(train_path)):
    name = re.findall('2018\d{4}',train_path[i])
    day.add(name[0])

day = list(day)
day.sort()

each_day_path = []
for each_day in day:
    sub_path = []
    for each_train_path in train_path:
        if each_day in each_train_path:
            sub_path.append(each_train_path)
    each_day_path.append(sub_path)

fields = ['track_id', 'duration', 'release_year', 'us_popularity_estimate',
       'acousticness', 'beat_strength', 'bounciness', 'danceability',
       'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mechanism', 'organism', 'speechiness', 'tempo',
       'time_signature', 'valence', 'acoustic_vector_0', 'acoustic_vector_1',
       'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
       'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7', 'count',
       'se_position_avg', 'skip_1_avg', 'skip_2_avg', 'skip_3_avg',
       'not_skip_avg']

dtypes = {
'track_id'          :        'object',
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
'count'             :        'int32',
'se_position_avg'   :        'float16',
'skip_1_avg'        :        'float16',
'skip_2_avg'        :        'float16',
'skip_3_avg'        :        'float16',
'not_skip_avg'      :        'float16'}

df_track = pd.read_csv('track_merge/track_update.csv',usecols = fields,dtype=dtypes)

fields = ['session_id','session_position','session_length','track_id_clean','skip_2']
dtypes = train_dtypes = {
'session_id'      :'object',
'session_position':'int8',
'session_length'  :'int8',
'track_id_clean'  :'object',
'skip_2'          :'int8'
}

for i in tqdm(range(0,66)):

    print(day[i])

    df_all = pd.DataFrame()

    for each_path in each_day_path[i]:
        df = pd.read_csv(each_path,dtype = dtypes,usecols=fields)
        df_all = pd.concat([df_all,df])

    df_all.reset_index(inplace=True,drop=True)
    df_all.rename(columns= {'track_id_clean':'track_id'}, inplace=True)

    df_all = pd.merge(df_all, df_track, on=['track_id'],how='left')
    df_all.drop(['session_id','track_id'],axis=1,inplace=True)

    filename = 'tranining_lstm/'+str(day[i])+'.csv'
    df_all.to_csv(filename,index=False)

    del df_all;
    gc.collect()


print("--- %s mins ---" % ((time.time() - start_time)/60))
