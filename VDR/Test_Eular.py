import numpy as np
import csv

DIR = r"D:\研究所\重点研究计划\data\0822"

datas = np.load(DIR + r'\datas.npy')
print(datas)
last_time = 0;
temp = []
for i in range(datas.shape[0]):
    if last_time != datas[i][29]:
        last_time = datas[i][29]
        temp.append(datas[i])
temp = np.array(temp)
datas = temp
Eular_by_Q = np.arctan2(2 * (datas[:, 16] * datas[:, 19] + datas[:, 17] * datas[:, 18]),
                        1 - 2 * (np.power(datas[:, 18], 2) + np.power(datas[:, 19], 2))) * 180 / np.pi
Eular_by_GNSS = datas[:, 25]

error = np.abs(Eular_by_Q - Eular_by_GNSS)

error_csv = csv.writer(open('eular_error.csv', 'w', newline=''))
for i in range(error.shape[0]):
    error_csv.writerow(str(error[i]))
print(Eular_by_Q)
