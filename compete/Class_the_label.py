import numpy as np
import os
import csv

DIR = 'D:\\研究所\\比赛\\train\\data_sorted_dealed\\Label\\'
label = ['still', 'walk', 'run', 'bike', 'car', 'bus', 'train', 'bus']
result = []
for i in range(658):
    print(i)
    temp = np.loadtxt(DIR + 'Label_' + str(i) + '.txt')
    this_label = label[int(np.mean(temp)) - 1]
    if i > 0:
        if this_label == result[-1][0]:
            result[-1][1] += int(temp.shape[0])
            continue
    this_len = int(temp.shape[0])
    result.append([this_label, this_len])

csvFile = open('result.csv','w',newline='')
writer = csv.writer(csvFile)
m = len(result)
for i in range(m):
    writer.writerow(result[i])
csvFile.close()