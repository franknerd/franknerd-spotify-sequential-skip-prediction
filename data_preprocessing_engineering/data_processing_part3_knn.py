import glob
import pandas as pd
import gc
from tqdm import tqdm
import pathlib
import time
import re

start_time = time.time()

''' import files '''
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

#print('')
#print('- - - - - type the number of files you want to process - - - - - ')
#start = int(input('From(include): '))
#end = int(input('To(include): '))

start = 1
#end = 1
end = len(train_path)

date_of_name = []
for i in range(len(train_path)):
    name = re.findall('2018\d{4}',train_path[i])
    date_of_name.append(name[0])


''' make directories '''
# make a directory if there are no directories
pathlib.Path('train_for_model_knn').mkdir(parents=True, exist_ok=True)
# make a directory if there are no directories
pathlib.Path('test_for_model_knn').mkdir(parents=True, exist_ok=True)


''' import track '''
dtypes = {
'track_id'          :        'object',
'count'             :        'int32',
'skip_2_avg_knn'    :        'float16',
}
track_fields = ['track_id','count','skip_2_avg_knn']
track_feature = pd.read_csv('track_merge/track_update_knn.csv',dtype=dtypes,usecols=track_fields)

''' import dataset '''
train_dtypes = {
'session_id'      :'object',
'session_position':'int8',
'session_length'  :'int8',
'track_id_clean'  :'object',
'skip_2'          :'int8'
}
train_fields = ['session_id','session_position',
                'session_length','track_id_clean','skip_2']

test_dtypes = {
'session_id'      :'object',
'session_position':'int8',
'session_length'  :'int8',
'track_id_clean'  :'object',
}

for i in tqdm(range(start-1,end,1)):

    # import files
    each_train = pd.read_csv(train_path[i],dtype=train_dtypes,usecols=train_fields)
    each_test = pd.read_csv(test_path[i],dtype=test_dtypes)

    # rename colunm names
    each_train.rename(columns={'track_id_clean':'track_id'}, inplace=True)
    each_test.rename(columns= {'track_id_clean':'track_id'}, inplace=True)

    # col skip_2_first_half
    # the mean of skip_2 proportion for the first half in each session
    first_half_skip_mean = each_train.groupby('session_id')[['skip_2']].mean()
    first_half_skip_mean.reset_index(inplace=True)
    first_half_skip_mean.rename(columns={'skip_2':'skip_2_first_half'}, inplace=True)

    each_train = pd.merge(each_train, first_half_skip_mean, on=['session_id'],how='left')
    each_test  = pd.merge(each_test, first_half_skip_mean, on=['session_id'],how='left')

    # merge track features to each row (energy', 'skip_2_avg', 'seekfwd_avg_log')
    each_train = pd.merge(each_train, track_feature, on=['track_id'],how='left')
    each_test  = pd.merge(each_test, track_feature, on=['track_id'],how='left')

    # del columns
    each_train.drop(['session_id','track_id'],axis=1,inplace=True)
    each_test.drop(['session_id','track_id'],axis=1,inplace=True)

    # save files
    train_for_model = 'train_for_model_knn/'+date_of_name[i]+'.csv'
    test_for_model  = 'test_for_model_knn/'+date_of_name[i]+'.csv'
    each_train.to_csv(train_for_model, index=False)
    each_test.to_csv(test_for_model, index=False)

    del each_train; del each_test; del first_half_skip_mean
    gc.collect()

print("--- %s mins ---" % ((time.time() - start_time)/60))
