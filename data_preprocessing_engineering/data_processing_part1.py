import glob
import pandas as pd
import gc
from tqdm import tqdm
import time
import os.path


start_time = time.time()

''' select files '''
#train_path = glob.glob('training_set/*')
train_path = glob.glob('test_set/*')
train_path = [ i for i in train_path if 'prehistory' in i ]
#train_path = glob.glob('sub_sample_1/*')

if len(train_path)>0:
    train_path.sort() # sorted
    for each_file in train_path:
        print(each_file)
    print('the number of train files:',len(train_path))
else:
    print('cannot find any file!')
    quit()

#print('')
#print('- - - - - type the number of files you want to process - - - - - ')
start = 1
end = len(train_path)

''' import track '''
fields = ['track_id']
df_track = pd.read_csv('track_merge/track_all.csv',usecols=fields)
df_track_id = df_track['track_id'].tolist()

''' each file'''
for i in tqdm(range(start-1,end,1)):

    dtypes = {
    'track_id_clean'    :'object',
    'session_position'  :'int8',
    'skip_1'            :'int8',
    'skip_2'            :'int8',
    'skip_3'            :'int8',
    'not_skipped'       :'int8',
    'context_switch'    :'int8',
    'no_pause_before_play'   :'int8',
    'short_pause_before_play':'int8',
    'long_pause_before_play' :'int8',
    'hist_user_behavior_n_seekfwd'  :'int16',
    'hist_user_behavior_n_seekback' :'int16',
    'hist_user_behavior_is_shuffle' :'int16',
    'premium'                       :'int8'
    }

    fields = ['track_id_clean','session_position','skip_1','skip_2',
    'skip_3','not_skipped','context_switch','no_pause_before_play',
    'short_pause_before_play','long_pause_before_play',
    'hist_user_behavior_n_seekfwd','hist_user_behavior_n_seekback',
    'hist_user_behavior_is_shuffle','premium']
    df_train = pd.read_csv(train_path[i], dtype=dtypes,usecols=fields)

    # generate some features
    df_sta = df_train.groupby('track_id_clean').agg({
                                            'track_id_clean':'count',
                                            'session_position':'sum',
                                            'skip_1':'sum',
                                            'skip_2':'sum',
                                            'skip_3':'sum',
                                            'not_skipped':'sum',
                                            'context_switch':'sum',
                                            'no_pause_before_play':'sum',
                                            'short_pause_before_play':'sum',
                                            'long_pause_before_play':'sum',
                                            'hist_user_behavior_n_seekfwd':'sum',
                                            'hist_user_behavior_n_seekback':'sum',
                                            'hist_user_behavior_is_shuffle':'sum',
                                            'premium':'sum'
                                            })

    df_sta.rename(columns={'track_id_clean':'count'}, inplace=True)
    df_sta.reset_index(inplace=True)
    df_sta.rename(columns={'track_id_clean':'track_id'}, inplace=True)

    del df_train;
    gc.collect()

    if os.path.isfile('track_merge/track_sta.csv') == False:
        df_sta.to_csv('track_merge/track_sta.csv',index=False)

        del df_sta;
        gc.collect()

    else:
        track_sta_original = pd.read_csv('track_merge/track_sta.csv')

        temp = pd.concat([df_sta, track_sta_original], ignore_index=True)
        temp.reset_index(drop=True,inplace=True)

        track_sta_new = temp.groupby('track_id').sum()
        track_sta_new.reset_index(inplace=True)
        track_sta_new.to_csv('track_merge/track_sta.csv',index=False)

        del track_sta_original; del temp; del track_sta_new
        gc.collect()

print("--- %s mins ---" % ((time.time() - start_time)/60))
