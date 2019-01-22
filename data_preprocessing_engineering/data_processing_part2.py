import pandas as pd
import time
import numpy as np

start_time = time.time()

'''import two datafrme'''
# dtypes = {
# 'track_id':'object',
# 'energy':'float16'
# }
#
# fields = ['track_id','energy']

#track_feature = pd.read_csv('track_merge/track_all.csv',dtype=dtypes,usecols=fields)

track_feature = pd.read_csv('track_merge/track_all.csv')
new_feature   = pd.read_csv('track_merge/track_sta.csv')

'''merge them'''
track_feature_all = pd.merge(track_feature,new_feature,on=['track_id'],how='left')

# avg
track_feature_all['se_position_avg'] = track_feature_all['session_position']/track_feature_all['count']
track_feature_all['skip_1_avg'] = track_feature_all['skip_1']/track_feature_all['count']
track_feature_all['skip_2_avg'] = track_feature_all['skip_2']/track_feature_all['count']
track_feature_all['skip_3_avg'] = track_feature_all['skip_3']/track_feature_all['count']
track_feature_all['not_skip_avg'] = track_feature_all['not_skipped']/track_feature_all['count']
track_feature_all['context_switch_avg'] = track_feature_all['context_switch']/track_feature_all['count']
track_feature_all['no_pause_bf_play_avg'] = track_feature_all['no_pause_before_play']/track_feature_all['count']
track_feature_all['short_pause_bf_play_avg'] = track_feature_all['short_pause_before_play']/track_feature_all['count']
track_feature_all['long_pause_bf_play_avg'] = track_feature_all['long_pause_before_play']/track_feature_all['count']
track_feature_all['hist_user_behavior_n_seekfwd_avg'] = track_feature_all['hist_user_behavior_n_seekfwd']/track_feature_all['count']
track_feature_all['hist_user_behavior_n_seekback_avg'] = track_feature_all['hist_user_behavior_n_seekback']/track_feature_all['count']
track_feature_all['hist_user_behavior_is_shuffle_avg'] = track_feature_all['hist_user_behavior_is_shuffle']/track_feature_all['count']
track_feature_all['premium_avg'] = track_feature_all['premium']/track_feature_all['count']

# log
track_feature_all['seekfwd_avg_log'] = np.log((track_feature_all['hist_user_behavior_n_seekfwd']/track_feature_all['count'])+1)

# drop
track_feature_all.drop(['session_position','skip_1','skip_2',
'skip_3','not_skipped','context_switch','no_pause_before_play',
'short_pause_before_play','long_pause_before_play',
'hist_user_behavior_n_seekfwd','hist_user_behavior_n_seekback',
'hist_user_behavior_is_shuffle','premium','mode'],axis=1,inplace=True)

# print and check nul
print('null')
print(track_feature_all.isnull().sum())
track_feature_all.fillna(0.0,inplace=True)

# save the file
track_feature_all.to_csv('track_merge/track_update.csv',index=False)

print("--- %s mins ---" % ((time.time() - start_time)/60))
