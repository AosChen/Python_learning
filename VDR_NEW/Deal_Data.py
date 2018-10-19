import numpy as np
from sklearn import preprocessing
import csv
import re
import time
import os
import pandas as pd
import math
import random


# 根据经纬度计算距离
def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = La1 * np.pi / 180.0
    radLa2 = La2 * np.pi / 180.0
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * math.asin(math.sqrt(
        math.pow(math.sin(deltaLa / 2), 2) + math.cos(radLa1) * math.cos(radLa2) * math.pow(math.sin(deltaLo / 2),
                                                                                            2))) * Er


def NMEA2LL(nmea_data):
    if nmea_data == '':
        return nmea_data
    data = float(nmea_data) / 100
    pointfront = int(data)
    pointback = (data - pointfront) * 100
    pointback *= 0.0166667
    data = pointfront + pointback
    return str(data)


def getposition(v, a):
    return v * 0.01 + 0.5 * a * 0.01 * 0.01


# 文件夹等信息静态常量
DIR = r"D:\研究所\重点研究计划\data\1015"
STATIC_FILE_NAME = r"\20181015155859_HUAWEI NXT-AL10__100Hz"
DEAL_DIR = r"\dealed_file"
DATE = r'2018-10-15 '

# 各个传感器属性名，根据该list的顺序进行存储，不要改值，修改后RLacc的计算也需要对应改变
attrs = ['acc', 'lacc', 'gacc', 'gyo', 'mag', 'q', 'pre']
Project_Start_TIME = 0
# 创建目标文件夹
if not os.path.exists(DIR + DEAL_DIR):
    os.mkdir(DIR + DEAL_DIR)

# 1.处理RTK数据信息
FILE = DIR + r'\20181015.dat'
csv_read = csv.reader(open(FILE, 'r', encoding='UTF-8'))
GNGGA = []
for row in csv_read:
    if 'GGA' in row[0]:
        UTC_TIME = row[1].split('.')
        HMS = str(int(UTC_TIME[0]) + 80000)
        MS = int(UTC_TIME[1]) / 100
        HMS = ':'.join(re.findall(r'.{2}', HMS))
        TIME = str(round((time.mktime(time.strptime(DATE + HMS, "%Y-%m-%d %H:%M:%S")) + MS) * 1000))
        Longitude = NMEA2LL(row[4])
        if row[5] == 'W':
            Longitude = str(float(Longitude) * -1)
        Latitude = NMEA2LL(row[2])
        if row[3] == 'S':
            Latitude = str(float(Latitude) * -1)
        Height = row[9]
        Mode = row[6]
        HDOP = row[8]
        if Project_Start_TIME == 0:
            Project_Start_TIME = int(TIME)
        TIME = str(int(TIME) - Project_Start_TIME)
        GNGGA.append([TIME, Longitude, Latitude, Height, HDOP, Mode, row[1]])
csv_writer = csv.writer(open(DIR + DEAL_DIR + r'\RTK_std.csv', 'w', newline=''))
for i in GNGGA:
    csv_writer.writerow(i)
rtk_pos = csv.writer(open(DIR + DEAL_DIR + r'\RTK_pos.csv', 'w', newline=''))
rtk_ll = []
for i in GNGGA:
    if float(i[6].split('.')[1]) != 0:
        continue
    rtk_ll.append([i[6], float(i[1]), float(i[2])])
for i in range(len(rtk_ll) - 1):
    front = DATE + (':'.join(re.findall(r'.{2}', rtk_ll[i][0].split('.')[0])))
    back = DATE + (':'.join(re.findall(r'.{2}', rtk_ll[i + 1][0].split('.')[0])))
    front = time.mktime(time.strptime(front, "%Y-%m-%d %H:%M:%S"))
    back = time.mktime(time.strptime(back, "%Y-%m-%d %H:%M:%S"))
    if back - front != 1:
        continue
    pE, pL = 1, 1
    if rtk_ll[i + 1][1] - rtk_ll[i][1] < 0:
        pE = -1
    if rtk_ll[i + 1][2] - rtk_ll[i][2] < 0:
        pL = -1
    rtk_pos.writerow([float(rtk_ll[i + 1][0]),
                      distance_by_LoLa(rtk_ll[i + 1][1], rtk_ll[i + 1][2], rtk_ll[i][1], rtk_ll[i + 1][2]) * pE,
                      distance_by_LoLa(rtk_ll[i + 1][1], rtk_ll[i + 1][2], rtk_ll[i + 1][1], rtk_ll[i][2]) * pL])

print('RTK数据处理完毕')

# 2.处理GPS的信息
start_UTC = 0
ngi_file_dealed = open(DIR + DEAL_DIR + r'\ngi.csv', 'w', newline='')
with open(DIR + STATIC_FILE_NAME + '_ngi.csv', 'r') as ngi_file:
    ngi_csv = csv.reader(ngi_file)
    ngi_csv_dealed = csv.writer(ngi_file_dealed)
    for row in ngi_csv:
        if len(row) != 10:
            continue
        if start_UTC != float(row[9]):
            row[0] = str(int(row[0]) - Project_Start_TIME)
            ngi_csv_dealed.writerow(row)
            start_UTC = float(row[9])
# gps信息写入新文件
ngi_file_dealed.close()
gps_pos = csv.writer(open(DIR + DEAL_DIR + r'\NGI_pos.csv', 'w', newline=''))
ngi_csv_dealed = csv.reader(open(DIR + DEAL_DIR + r'\ngi.csv', 'r'))
ngi_ll, ngi_pos = [], {}
for i in ngi_csv_dealed:
    if len(str(i[9]).split('.')[0]) < 6:
        i[9] = '0' + str(i[9]).split('.')[0]
    ngi_ll.append([i[9], float(i[1]), float(i[2])])
for i in range(len(ngi_ll) - 1):
    front = DATE + (':'.join(re.findall(r'.{2}', ngi_ll[i][0].split('.')[0])))
    back = DATE + (':'.join(re.findall(r'.{2}', ngi_ll[i + 1][0].split('.')[0])))
    front = time.mktime(time.strptime(front, "%Y-%m-%d %H:%M:%S"))
    back = time.mktime(time.strptime(back, "%Y-%m-%d %H:%M:%S"))
    if back - front != 1:
        continue
    pE, pL = 1, 1
    if ngi_ll[i + 1][1] - ngi_ll[i][1] < 0:
        pE = -1
    if ngi_ll[i + 1][2] - ngi_ll[i][2] < 0:
        pL = -1
    ngi_pos[float(ngi_ll[i + 1][0])] = [
        distance_by_LoLa(ngi_ll[i + 1][1], ngi_ll[i + 1][2], ngi_ll[i][1], ngi_ll[i + 1][2]) * pE,
        distance_by_LoLa(ngi_ll[i + 1][1], ngi_ll[i + 1][2], ngi_ll[i + 1][1], ngi_ll[i][2]) * pL
    ]
    gps_pos.writerow(ngi_pos)
print('GPS数据处理完毕')

# 3.处理其余传感器文件，以acc时间为基准
acc_file = pd.read_csv(open(DIR + STATIC_FILE_NAME + '_acc.csv'), names=list('tabc'))
t_std = np.array(list(map(int, acc_file[list('t')].values))) - Project_Start_TIME
last_data = [t_std]

for attr in attrs:
    size = 3
    if attr == 'q':
        size = 4
    elif attr == 'pre':
        size = 1
    list_name = 't'
    for i in range(size):
        list_name += str(i)
    file_read = pd.read_csv(open(DIR + STATIC_FILE_NAME + "_" + attr + ".csv"), names=list(list_name))
    for i in range(size):
        temp = np.interp(t_std,
                         np.array(list(map(int, file_read[list('t')].values))) - Project_Start_TIME,
                         np.array(list(map(float, file_read[list(str(i))].values)))
                         )
        if attr == 'pre':
            minmax = preprocessing.MinMaxScaler()
            temp = minmax.fit_transform(temp.reshape(-1, 1))
            temp = temp.flatten()
        last_data.append(temp)

new_file_data = np.array(last_data)
new_file_data = np.transpose(new_file_data)

csv_writer = csv.writer(open(DIR + DEAL_DIR + '\\datas.csv', 'w', newline=''))
for i in range(new_file_data.shape[0]):
    csv_writer.writerow(new_file_data[i])
# 传感器文件处理完毕
print('INS数据处理完毕')

# 4.进行数据的整合处理，即找到两个GPS时间间隔内的传感器数据，同时对GPS数据进行计算处理
Hidden_States, Inputs, Outputs, V_DEAL, P_INS_Target = [], [], [], [], []
isInitdata = True
last_time, lastE, lastL, last_height = 0, 0, 0, 0

csv_reader = csv.reader(open(DIR + DEAL_DIR + '\\ngi.csv', 'r'))
NGI = []
for row in csv_reader:
    NGI.append(list(map(float, row)))
NGI = np.array(NGI)
minmax = preprocessing.MinMaxScaler()
NGI_V_DEAL = minmax.fit_transform(NGI[:, 4].reshape(-1, 1))

csv_reader = csv.reader(open(DIR + DEAL_DIR + '\\datas.csv', 'r'))
DATA = []
for row in csv_reader:
    DATA.append(list(map(float, row)))
DATA = np.array(DATA)

j = 0
for i in range(NGI.shape[0] - 1):
    Input = []
    if NGI[i][8] > 3.0 or NGI[i + 1][8] > 3.0 \
            or (NGI[i][1] == 0 and NGI[i][2] == 0) or (NGI[i + 1][1] == 0 and NGI[i + 1][2] == 0) \
            or (NGI[i][4] == 0 and NGI[i][5] == 0) or (NGI[i + 1][4] == 0 and NGI[i + 1][5] == 0):
        isInitdata = True
        continue
    if isInitdata:
        last_time, lastE, lastL, last_height = NGI[i][9], NGI[i][1], NGI[i][2], NGI[i][3]
        isInitdata = False
        continue
    Hidden_States.append(
        [NGI[i][9], 0, 0, 0, math.cos(NGI[i][5] * math.pi / 180), math.sin(NGI[i][5] * math.pi / 180), NGI[i][4],
         NGI[i][3] - last_height])
    V_DEAL.append([math.cos(NGI[i][5] * math.pi / 180), math.sin(NGI[i][5] * math.pi / 180), NGI_V_DEAL[i][0]])
    last_time, lastE, lastL, last_height = NGI[i][9], NGI[i][1], NGI[i][2], NGI[i][3]
    pE, pL = 1, 1
    if NGI[i + 1][1] - lastE < 0:
        pE = -1
    if NGI[i + 1][2] - lastL < 0:
        pL = -1
    Outputs.append(
        [NGI[i + 1][9], distance_by_LoLa(NGI[i + 1][1], NGI[i + 1][2], lastE, NGI[i + 1][2]) * pE,
         distance_by_LoLa(NGI[i + 1][1], NGI[i + 1][2], NGI[i + 1][1], lastL) * pL,
         NGI[i + 1][3] - last_height, math.cos(NGI[i + 1][5] * math.pi / 180),
         math.sin(NGI[i][5] * math.pi / 180), NGI[i + 1][4], NGI[i + 1][3] - last_height]
    )
    TimeStep_start, TimeStep_stop = NGI[i][0], NGI[i + 1][0]
    while True:
        if DATA[j][0] > TimeStep_stop:
            break
        else:
            if DATA[j][0] >= TimeStep_start:
                Input.append([DATA[j][4], DATA[j][5], DATA[j][6], DATA[j][16], DATA[j][17], DATA[j][18], DATA[j][19],
                              DATA[j][20], DATA[j][10], DATA[j][11], DATA[j][12], Hidden_States[-1][6] / 100,
                              V_DEAL[-1][0], V_DEAL[-1][1]])
            j += 1
    while len(Input) < 100:
        Input.append(Input[random.randint(0, len(Input) - 1)])
    while len(Input) > 100:
        Input.pop(random.randint(0, len(Input) - 1))
    px_ins, py_ins = 0, 0
    for temp_data in Input:
        q0, q1, q2, q3 = temp_data[3], temp_data[4], temp_data[5], temp_data[6]
        Matrix_R = np.array(
            [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
             [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
             [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
        )
        Matrix_A = temp_data[0:3]
        Matrix_A = np.dot(Matrix_R, Matrix_A).tolist()
        px_ins += getposition(Hidden_States[-1][6] * Hidden_States[-1][4], Matrix_A[0])
        py_ins += getposition(Hidden_States[-1][6] * Hidden_States[-1][5], Matrix_A[1])

    Input = np.array(Input)
    temp = np.transpose(np.array([px_ins, py_ins]).repeat(100).reshape(-1, 100))
    Input = np.hstack([Input, temp])
    Inputs.append(Input)
    P_INS_Target.append([ngi_pos[Outputs[-1][0]][0] - px_ins, ngi_pos[Outputs[-1][0]][1] - py_ins,
                         (Outputs[-1][6] - Hidden_States[-1][6]) / 100,
                         Outputs[-1][4] - Hidden_States[-1][4], Outputs[-1][5] - Hidden_States[-1][5]]
                        )

# output_csv = csv.writer(open(DIR + DEAL_DIR + '\\Outputs.csv', 'w', newline=''))
# for i in range(len(Outputs)):
#     output_csv.writerow(Outputs[i])

Hidden_States = np.array(Hidden_States)
Outputs = np.array(Outputs)
Inputs = np.array(Inputs)
P_INS_Target = np.array(P_INS_Target)
V_DEAL = np.array(V_DEAL)

np.save(DIR + DEAL_DIR + '\\Hidden_States.npy', Hidden_States)
np.save(DIR + DEAL_DIR + '\\Outputs.npy', Outputs)
np.save(DIR + DEAL_DIR + '\\Inputs.npy', Inputs)
np.save(DIR + DEAL_DIR + '\\P_INS_Target.npy', P_INS_Target)
np.save(DIR + DEAL_DIR + '\\V_Deal.npy', V_DEAL)

print('模型输入数据文件处理完毕')
