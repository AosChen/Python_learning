import numpy as np
from sklearn import preprocessing
import csv
import math

DIR = r"D:\研究所\重点研究计划\data\0822"
FILE_NAME = r"\20180821201504_HUAWEI NXT-AL10__100Hz.csv"


def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = La1 * np.pi / 180.0
    radLa2 = La2 * np.pi / 180.0
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * math.asin(math.sqrt(
        math.pow(math.sin(deltaLa / 2), 2) + math.cos(radLa1) * math.cos(radLa2) * math.pow(math.sin(deltaLo / 2),
                                                                                            2))) * Er


csv_reader = csv.reader(open(DIR + FILE_NAME, 'r'))
index = 0

attr = []
datas = []
for row in csv_reader:
    if index == 0:
        print(row)
        attr = row
        index += 1
    else:
        datas.append(list(map(float, row)))

datas = np.array(datas)
np.save(DIR + r'\datas.npy', datas)

datas = np.load(DIR + r'\datas.npy')
print(datas)

pre = datas[:, 20]
preprocessing.MinMaxScaler(pre)
datas[:, 20] = pre

datas_size = datas.shape[0]
datas_step = 100

Hidden_States, Inputs, Outputs = [], [], []
Hidden_State, Input, Output = [], [], []
# Hidden_State包括：三轴位移量，GNSS航向角，GNSS速度，GNSS对天速度，对天速度根据高程计算

# 需要记录一个状态变量来保证，当前记录的数据，是作为要写入的数据，还是作为起始点数据
# 需要反复记录起始点数据的原因在于，可能出现某一个时间段，GNSS数据无法使用，重新获取的数据是不能被使用的
isInitdata = True
last_time, lastE, lastL, last_height = 0, 0, 0, 0

for i in range(datas_size):
    if datas[i][28] > 3.0:
        Hidden_State, Input, Output = [], [], []
        isInitdata = True
        continue
    if isInitdata:
        if datas[i][29] != last_time:
            if last_time == 0:
                last_time, lastE, lastL, last_height = datas[i][29], datas[i][21], datas[i][22], datas[i][23]
            else:
                isInitdata = False
                Hidden_State = [0, 0, 0, math.cos(datas[i][25] * math.pi / 180), math.sin(datas[i][25] * math.pi / 180),
                                datas[i][24], datas[i][23] - last_height]
                last_time, lastE, lastL, last_height = datas[i][29], datas[i][21], datas[i][22], datas[i][23]
                Input.append(
                    [datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19],
                     datas[i][20]])
    else:
        if datas[i][29] == last_time:
            if len(Input) < 100:
                Input.append(
                    [datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19],
                     datas[i][20]])
        else:
            while len(Input) < 100:
                Input.append(Input[-1])
            pE, pL = 1, 1
            if datas[i][21] - lastE < 0:
                pE = -1
            if datas[i][22] - lastL < 0:
                pL = -1
            Output = [distance_by_LoLa(datas[i][21], datas[i][22], lastE, datas[i][22]) * pE,
                      distance_by_LoLa(datas[i][21], datas[i][22], datas[i][21], lastL) * pL,
                      datas[i][23] - last_height, math.cos(datas[i][25] * math.pi / 180),
                      math.sin(datas[i][25] * math.pi / 180), datas[i][24], datas[i][23] - last_height]
            Hidden_States.append(Hidden_State)
            Inputs.append(Input)
            Outputs.append(Output)
            Hidden_State, Input, Output = [], [], []
            Hidden_State = [0, 0, 0, math.cos(datas[i][25] * math.pi / 180),
                            math.sin(datas[i][25] * math.pi / 180), datas[i][24], datas[i][23] - last_height]
            Input.append(
                [datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19],
                 datas[i][20]])
            last_time, lastE, lastL, last_height = datas[i][29], datas[i][21], datas[i][22], datas[i][23]

Hidden_States = np.array(Hidden_States)
Inputs = np.array(Inputs)
Outputs = np.array(Outputs)

np.save(DIR + r'\Hidden_States.npy', Hidden_States)
np.save(DIR + r'\Inputs.npy', Inputs)
np.save(DIR + r'\Outputs.npy', Outputs)
