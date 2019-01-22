
import glob
import sys

path = sys.argv[1]

'''merge all output into one'''

path_glob = str(path) + '/*'
out_path = glob.glob(path_glob)
out_path.sort()

total =''
for each_file in out_path:
    with open(each_file, 'r') as file:
        data =file.read()
        total += data

path_file = str(path) +'/submission.txt'
output_file = open(path_file, 'w')
output_file.write(total)
output_file.close()
