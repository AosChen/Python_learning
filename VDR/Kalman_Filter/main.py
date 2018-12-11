import csv
import copy
import cdf
import numpy as np
from VDR.Kalman_Filter.IntegratedLocation import IntegratedLocation
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

print('数据处理完毕，共有数据' + str(len(datas)) + '行')

time_start = datas[int(0.8*len(datas))][-1]
time_stop = datas[-1][-1]
start_index, stop_index = -1, -1
step = 100 * 5
Errors = []
for i in range(len(datas)):
    if datas[i][-1] == time_start and start_index == -1:
        start_index = i
    if datas[i][-1] == time_stop and stop_index == -1:
        stop_index = i

il = IntegratedLocation(samplePeriod=0.01, smooth_size=20)
last_UTC = 0

# 前置训练
for i in range(start_index):
    Changed = False
    if datas[i][-1] != last_UTC:
        last_UTC = datas[i][-1]
        Changed = True
    result = il.deal_data(datas[i], Changed)
    # if datas[i][-9] == 0:
    #     continue
    # print("真值经纬度：" + str(datas[i][-9]) + "," + str(datas[i][-8]))
    # print("计算经纬度：" + str(result[0]) + "," + str(result[1]))
    # print("距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], result[1])))
    # print("东向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], datas[i][-8])))
    # print("北向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], datas[i][-9], result[1])))
    # Errors.append([distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], result[1]),
    #                distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], datas[i][-8]),
    #                distance_by_LoLa(datas[i][-9], datas[i][-8], datas[i][-9], result[1])])
    # print("------------------------------------------------------------------------------------------")

# 测试部分
while start_index < stop_index - step:
    temp = copy.deepcopy(il)
    for i in range(start_index, start_index + step + 1):
        Changed = False
        result = temp.deal_data(datas[i], Changed)
    if datas[i][-9] != 0:
        print("真值经纬度：" + str(datas[start_index + step][-9]) + "," + str(datas[start_index + step][-8]))
        print("计算经纬度：" + str(result[0]) + "," + str(result[1]))
        print("距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], result[1])))
        print("东向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], datas[i][-8])))
        print("北向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], datas[i][-9], result[1])))
        Errors.append([distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], result[1]),
                       distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], datas[i][-8]),
                       distance_by_LoLa(datas[i][-9], datas[i][-8], datas[i][-9], result[1])])
        print("------------------------------------------------------------------------------------------")
    temp_index = -1
    for i in range(start_index, len(datas)):
        if datas[i][-1] != datas[start_index][-1]:
            temp_index = i
            break
    for i in range(start_index, temp_index):
        Changed = False
        if datas[i][-1] != last_UTC:
            last_UTC = datas[i][-1]
            Changed = True
        result = il.deal_data(datas[i], Changed)
    start_index = temp_index

# while start_index < stop_index:
#     il = IntegratedLocation(samplePeriod=0.01, smooth_size=20)
#     last_UTC = 0
#     for i in range(len(datas)):
#         Changed = False
#         if datas[i][-1] != last_UTC:
#             last_UTC = datas[i][-1]
#             Changed = True
#         if start_index < i <= start_index + step:
#             Changed = False
#         result = il.deal_data(datas[i], Changed)
#         if i == start_index + step:
#             print("真值经纬度：" + str(datas[i][-9]) + "," + str(datas[i][-8]))
#             print("计算经纬度：" + str(result[0]) + "," + str(result[1]))
#             print("距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], result[1])))
#             print("东向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], result[0], datas[i][-8])))
#             print("北向距离误差：" + str(distance_by_LoLa(datas[i][-9], datas[i][-8], datas[i][-9], result[1])))
#             print("------------------------------------------------------------------------------------------")
#             break
#     for i in range(start_index, len(datas)):
#         if datas[i][-1] != datas[start_index][-1]:
#             start_index = i
#             break
for i in Errors:
    error_writer.writerow(i)

Errors = np.transpose(np.array(Errors)).tolist()

cdf.figure()
cdf.CDF_One_Print(Errors, ['error', 'east_error', 'north_error'], '5s GNSS outrage误差分布图')
cdf.show()
