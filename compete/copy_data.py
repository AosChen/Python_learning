import os
import numpy as np

lens = [76469, 100045, 32108, 96124, 82207, 75285, 50552, 45619]
max_len = max(lens)

for i in range(8):
    label = i + 1
    print('Dealing the Label' + str(label))
    temp = np.loadtxt('D:\\研究所\\比赛\\features\\old\\original_data_Label'+ str(label) + '_features.txt')
    print('Now the Label' + str(label) + ' shape is ', temp.shape)
    for j in range(max_len - lens[i]):
        index = np.random.choice(lens[i])
        temp = np.vstack((temp, temp[index]))
        print('Now the Label' + str(label) + ' shape is ', temp.shape)
    np.savetxt('D:\\研究所\\比赛\\features\\original_data_Label'+ str(label) + '_features.txt', temp)
    print('Dealing the Label' + str(label) + ' over!!!!!')
    print('==============================================')