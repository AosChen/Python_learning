import numpy as np
import os

DIR = 'D:\\研究所\\比赛\\train\\data_classed_by_label'
FILE_DIR = os.listdir(DIR)
for i in FILE_DIR:
    print(i)
for i in range(1, 9):
    Flag = 'Label' + str(i)
    print('==============================')
    temp = np.loadtxt(os.path.join(DIR, FILE_DIR[0]) + '\\' + FILE_DIR[0] + '_' + Flag + '.txt', dtype=np.float32)
    print('loading file ' + FILE_DIR[0] + '_' + Flag)
    print('now the temp\'s shape is ', temp.shape)
    for j in range(len(FILE_DIR)):
        if j == 0 or FILE_DIR[j] == 'Label':
            continue
        print('loading file ' + FILE_DIR[j] + '_' + Flag)
        temp = np.vstack((temp, np.loadtxt(os.path.join(DIR, FILE_DIR[j]) + '\\' + FILE_DIR[j] + '_' + Flag + '.txt', dtype=np.float32)))
        print('now the temp\'s shape is ', temp.shape)
    temp = temp.T
    print('now the temp\'s shape is ', temp.shape)
    np.savetxt(DIR + '\\' + Flag + '_concat.txt', temp)
    print('saving ' + Flag + ' over!!')
