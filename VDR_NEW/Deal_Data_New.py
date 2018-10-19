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


def getRMatrix(q0, q1, q2, q3, ax, ay, az):
    Matrix_R = np.array(
        [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
         [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
         [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
    )
    Matrix_A = np.dot(Matrix_R, [ax, ay, az]).tolist()
    return Matrix_A


# 文件夹等信息静态常量
DIR = r"D:\研究所\重点研究计划\data\1015"
STATIC_FILE_NAME = r"\20181015155859_HUAWEI NXT-AL10__100Hz"
DEAL_DIR = r"\dealed_file"
DATE = r'2018-10-15 '

# 各个传感器属性名，根据该list的顺序进行存储，不要改值，修改后RLacc的计算也需要对应改变
attrs = ['acc', 'lacc', 'gacc', 'gyo', 'mag', 'q', 'pre']
co_deleted = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 26, 33, 34, 35]
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
        if float(UTC_TIME[1]) != 0:
            continue
        TIME = UTC_TIME[0]
        Longitude = NMEA2LL(row[4])
        if row[5] == 'W':
            Longitude = float(Longitude) * -1
        Latitude = NMEA2LL(row[2])
        if row[3] == 'S':
            Latitude = float(Latitude) * -1
        Height = row[9]
        Mode = row[6]
        HDOP = row[8]
        GNGGA.append([TIME, float(Longitude), float(Latitude), float(Height), HDOP, Mode])
rtk_pos = csv.writer(open(DIR + DEAL_DIR + r'\RTK_pos.csv', 'w', newline=''))
for i in range(len(GNGGA) - 1):
    front = DATE + (':'.join(re.findall(r'.{2}', str(GNGGA[i][0]).split('.')[0])))
    back = DATE + (':'.join(re.findall(r'.{2}', str(GNGGA[i + 1][0]).split('.')[0])))
    front = time.mktime(time.strptime(front, "%Y-%m-%d %H:%M:%S"))
    back = time.mktime(time.strptime(back, "%Y-%m-%d %H:%M:%S"))
    if back - front != 1:
        continue
    pE, pL = 1, 1
    if GNGGA[i + 1][1] - GNGGA[i][1] < 0:
        pE = -1
    if GNGGA[i + 1][2] - GNGGA[i][2] < 0:
        pL = -1
    rtk_pos.writerow([float(GNGGA[i + 1][0]),
                      distance_by_LoLa(GNGGA[i + 1][1], GNGGA[i + 1][2], GNGGA[i][1], GNGGA[i + 1][2]) * pE,
                      distance_by_LoLa(GNGGA[i + 1][1], GNGGA[i + 1][2], GNGGA[i + 1][1], GNGGA[i][2]) * pL])
print('RTK数据处理完毕')

# 2.处理其余的文件
Data_reader = csv.reader(open(DIR + STATIC_FILE_NAME + '.csv', 'r'))
index = 0
Datas_Needed = []
for row in Data_reader:
    if index == 0:
        index += 1
        continue
    else:
        row = list(map(float, row))
        Datas_Needed.append(row)
Datas_Needed = np.array(Datas_Needed)
Datas_Needed = np.delete(Datas_Needed, co_deleted, axis=1)

# 气压计数值正则化并替换
minmax = preprocessing.MinMaxScaler()
pre = minmax.fit_transform(Datas_Needed[:, 10].reshape(-1, 1))
V_GPS = minmax.fit_transform(Datas_Needed[:, 14].reshape(-1, 1))
for i in range(Datas_Needed.shape[0]):
    Datas_Needed[i][10] = pre[i][0]

# 构建模型所需文件
Time_DEVIDE = []
last_UTC = 0
start, stop = 0, 0
for i in range(Datas_Needed.shape[0]):
    if Datas_Needed[i][-1] != last_UTC:
        if last_UTC == 0:
            last_UTC = Datas_Needed[i][-1]
            continue
        else:
            last_UTC = Datas_Needed[i][-1]
            stop = i
            Time_DEVIDE.append([start, stop])
            start = stop
Time_DEVIDE.pop(0)
for Time in Time_DEVIDE:
    Start_Time, Stop_Time = Time[0], Time[1]

