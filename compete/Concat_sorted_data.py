import os
import numpy as np

ROOT_DIR = 'D:\\研究所\\比赛\\train\\data_dealed'
AIM_DIR = 'D:\\研究所\\比赛\\train\\data_sorted_dealed'
Dir_Name = os.listdir(ROOT_DIR)



def openfile(data_type, index=0):
    return np.load(os.path.join(ROOT_DIR, data_type) + '\\' + data_type + '_' + str(index) + '.npy')


if not os.path.exists(AIM_DIR):
    os.makedirs(AIM_DIR)

split_index = np.loadtxt('D:\\研究所\\比赛\\train\\index.txt', dtype=np.int32)
split_index_i = 0

for j in Dir_Name:
    split_index_i = 0
    # if j == 'Label' or j == 'Acc_x':
    #     continue
    if not os.path.exists(os.path.join(AIM_DIR, j)):
        os.makedirs(os.path.join(AIM_DIR, j))
    temp = openfile(j, 0)
    how_many = 0
    for i in range(1, 16310):
        this_array = openfile(j, i)

        if split_index_i >= split_index.shape[0]:
            temp = np.hstack((temp, this_array))
            continue
        # 表示还没到这儿来
        if i < split_index[split_index_i][0]:
            temp = np.hstack((temp, this_array))
        else:
            this_row_index = [0]
            while split_index[split_index_i][0] == i:
                this_row_index.append(split_index[split_index_i][1])
                split_index_i += 1
                if split_index_i >= split_index.shape[0]:
                    break
            arrays = []
            for i in range(len(this_row_index) - 1):
                arrays.append(this_array[this_row_index[i]:this_row_index[i+1]])
            arrays.append(this_array[this_row_index[-1]:])
            temp = np.hstack((temp, arrays[0]))
            np.savetxt(os.path.join(AIM_DIR, j) + '\\' + j + '_' + str(how_many) + '.txt', temp)
            print('saving ' + j + '_' + str(how_many) + ' over')
            how_many += 1
            arrays.pop(0)
            while len(arrays) >= 2:
                np.savetxt(os.path.join(AIM_DIR, j) + '\\' + j + '_' + str(how_many) + '.txt', arrays[0])
                print('saving ' + j + '_' + str(how_many) + ' over')
                how_many += 1
                arrays.pop(0)
            temp = arrays[0]

    np.savetxt(os.path.join(AIM_DIR, j) + '\\' + j + '_' + str(how_many) + '.txt', temp)
    print('saving ' + j + '_' + str(how_many) + ' over')
