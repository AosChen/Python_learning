import csv
import copy
import cdf
import numpy as np
from VDR.Kalman_Filter.KF_NN.Calc import Calc
from VDR.tools import distance_by_LoLa

DIR = r'D:\研究所\重点研究计划\data\1015'
FILE = r'\20181015155859_HUAWEI NXT-AL10__100Hz - 副本.csv'
error = r'\error.csv'

STATIC_INDEX = 1000

File_reader = csv.reader(open(DIR + FILE, 'r'))
error_writer = csv.writer(open(DIR + error, 'w', newline=''))

index = -1
datas = []
for row in File_reader:
    if index == -1:
        index += 1
        continue
    row = list(map(float, row))
    if len(datas) == 0:
        datas.append([0 for _ in range(len(row))])
    if index < STATIC_INDEX:
        for i in range(20):
            datas[-1][i] += row[i]
        for i in range(-9, 0):
            datas[-1][i] = row[i]
    else:
        if index == STATIC_INDEX:
            for i in range(20):
                datas[-1][i] /= STATIC_INDEX
        datas.append(row)
    index += 1

index = -1
data_set = []
last_Time = 0
for data in datas:
    if index == -1:
        index += 1
        continue
    else:
        data = list(map(float, data))
        if data[-1] != last_Time:
            if last_Time == 0:
                data_set.append([data])
            else:
                data_set[-1].append(data)
                data_set.append([])
            last_Time = data[-1]
        else:
            data_set[-1].append(data)

print('数据处理完毕')
calc = Calc(samplePeriod=0.01, smooth_size=20)

Train_size = 0.8
outrage_step = 5
Errors = []


# Train
for i in range(int(len(data_set) * Train_size)):
    result = calc.deal_data(data_set[i])

# Test
for i in range(int(len(data_set) * Train_size), len(data_set) - 1 - outrage_step):
    temp = copy.deepcopy(calc)
    for j in range(outrage_step):
        result = temp.deal_data(data_set[i + j], do_NN=True)
    if data_set[i + j][-1][-9] != 0:
        print("真值经纬度：" + str(data_set[i + j][-1][-9]) + "," + str(data_set[i + j][-1][-8]))
        print("计算经纬度：" + str(result[0]) + "," + str(result[1]))
        print("距离误差：" + str(distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], result[0], result[1])))
        print("东向距离误差：" + str(distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], result[0], data_set[i + j][-1][-8])))
        print("北向距离误差：" + str(distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], data_set[i + j][-1][-9], result[1])))
        Errors.append([distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], result[0], result[1]),
                       distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], result[0], data_set[i + j][-1][-8]),
                       distance_by_LoLa(data_set[i + j][-1][-9], data_set[i + j][-1][-8], data_set[i + j][-1][-9], result[1])])
        print("------------------------------------------------------------------------------------------")
    calc.deal_data(data_set[i], do_NN=False)

for i in Errors:
    error_writer.writerow(i)
Errors = np.transpose(np.array(Errors)).tolist()
cdf.figure()
cdf.CDF_One_Print(Errors, ['error', 'east_error', 'north_error'], '5s GNSS outrage误差分布图')
cdf.show()



