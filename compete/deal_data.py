import os
import numpy as np

ROOT_DIR = 'D:\\研究所\\比赛\\train\\data_sorted_dealed'
AIM_DIR = 'D:\\研究所\\比赛\\train\\new_data'
Dir_Name = os.listdir(ROOT_DIR)

index = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

if not os.path.exists(AIM_DIR):
    os.makedirs(AIM_DIR)

for i in range(658):
    print('dealing the file ', str(i))
    temp = np.loadtxt(os.path.join(ROOT_DIR, Dir_Name[0]) + '\\' + Dir_Name[0] + '_' + str(i) + '.txt', dtype=np.float32)
    for Data_type in range(1, len(Dir_Name)):
        if Dir_Name[Data_type] == 'Label':
            continue
        temp = np.vstack(
            (temp, np.loadtxt(os.path.join(ROOT_DIR, Dir_Name[Data_type]) + '\\' + Dir_Name[Data_type] + '_' + str(i) + '.txt', dtype=np.float32)))
    temp = temp.T
    print('now the temp shape is ', temp.shape)
    mean = int(np.mean(np.loadtxt(os.path.join(ROOT_DIR, 'Label') + '\\Label_' + str(i) + '.txt')))
    print('mean is ', str(mean))
    np.savetxt(os.path.join(AIM_DIR, 'Label'+str(mean)) + '\\' + str(index[mean]) + '.txt', temp)
    index[mean] += 1
