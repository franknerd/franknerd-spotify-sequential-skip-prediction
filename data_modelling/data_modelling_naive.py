import glob
import pandas as pd
import time
from tqdm import tqdm
import re
import pathlib
import gc

start_time = time.time()

train_path = glob.glob('train_for_model/*')
test_path = glob.glob('test_for_model/*')

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


''' make directories '''
# make a directory if there are no directories
pathlib.Path('output_0102_v1').mkdir(parents=True, exist_ok=True)


train_fields = ['session_position','session_length','skip_2']
test_fields = ['session_position','session_length']


for i_file in tqdm(range(0,66,1)):
    train = pd.read_csv(train_path[i_file],usecols=train_fields)
    test = pd.read_csv(test_path[i_file],usecols=test_fields)

    output = ''

    train_pos = train['session_position'].tolist()
    train_length = train['session_length'].tolist()

    train_skip = train['skip_2'].tolist()

    for i in range(len(train_pos)):
        if train_pos[i] == train_length[i]//2:
            length = train_length[i]-train_pos[i]
            output += str(train_skip[i])*length
            output += '\n'

    '''save file'''
    output_path = 'output_0102_v1/'+date_of_name[i_file]+'.txt'
    output_file = open(output_path, 'w')
    output_file.write(output)
    output_file.close()

print("--- %s mins ---" % ((time.time() - start_time)/60))
