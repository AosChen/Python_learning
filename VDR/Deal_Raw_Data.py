import numpy as np
from sklearn import preprocessing
import csv
import math

DIR = r"D:\研究所\重点研究计划\data\0824"
FILE_NAME = r"\20180822184542_HUAWEI NXT-AL10__100Hz"


def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = La1 * np.pi / 180.0
    radLa2 = La2 * np.pi / 180.0
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * math.asin(math.sqrt(
        math.pow(math.sin(deltaLa / 2), 2) + math.cos(radLa1) * math.cos(radLa2) * math.pow(math.sin(deltaLo / 2),
                                                                                            2))) * Er


csv_reader = csv.reader(open(DIR + FILE_NAME + '.csv', 'r'))
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
np.save(DIR + FILE_NAME + r'_datas.npy', datas)

datas = np.load(DIR + FILE_NAME + r'_datas.npy')
print(datas)

pre = datas[:, 20]
minmax = preprocessing.MinMaxScaler()
pre = minmax.fit_transform(pre.reshape(-1, 1))
datas[:, 20] = pre.flatten()

datas_size = datas.shape[0]
datas_step = 100

Hidden_States, Inputs, Outputs, RLaccs = [], [], [], []
Hidden_State, Input, Output, RLacc = [], [], [], []
# Hidden_State包括：三轴位移量，GNSS航向角，GNSS速度，GNSS对天速度，对天速度根据高程计算

# 需要记录一个状态变量来保证，当前记录的数据，是作为要写入的数据，还是作为起始点数据
# 需要反复记录起始点数据的原因在于，可能出现某一个时间段，GNSS数据无法使用，重新获取的数据是不能被使用的
isInitdata = True
last_time, lastE, lastL, last_height = 0, 0, 0, 0

for i in range(datas_size):
    if datas[i][28] > 3.0:
        Hidden_State, Input, Output, RLacc = [], [], [], []
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
                q0, q1, q2, q3 = datas[i][16], datas[i][17], datas[i][18], datas[i][19]
                Matrix_R = np.array(
                    [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
                     [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                     [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])
                Matrix_A = np.array([[datas[i][4]], [datas[i][5]], [datas[i][6]]])
                Matrix_A = np.dot(Matrix_R, Matrix_A).tolist()
                RLacc.append([Matrix_A[0][0], Matrix_A[1][0], Matrix_A[2][0]])
    else:
        if datas[i][29] == last_time:
            if len(Input) < 100:
                Input.append(
                    [datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19],
                     datas[i][20]])
                q0, q1, q2, q3 = datas[i][16], datas[i][17], datas[i][18], datas[i][19]
                Matrix_R = np.array(
                    [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
                     [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                     [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])
                Matrix_A = np.array([[datas[i][4]], [datas[i][5]], [datas[i][6]]])
                Matrix_A = np.dot(Matrix_R, Matrix_A).tolist()
                RLacc.append([Matrix_A[0][0], Matrix_A[1][0], Matrix_A[2][0]])
        else:
            while len(Input) < 100:
                Input.append(Input[-1])
                RLacc.append(RLacc[-1])
            pE, pL = 1, 1
            if datas[i][21] - lastE < 0:
                pE = -1
            if datas[i][22] - lastL < 0:
                pL = -1
            Output = [distance_by_LoLa(datas[i][21], datas[i][22], lastE, datas[i][22]) * pE,
                      distance_by_LoLa(datas[i][21], datas[i][22], datas[i][21], lastL) * pL,
                      datas[i][23] - last_height, math.cos(datas[i][25] * math.pi / 180),
                      math.sin(datas[i][25] * math.pi / 180), datas[i][24], datas[i][23] - last_height,
                      datas[i][29]]
            Hidden_States.append(Hidden_State)
            Inputs.append(Input)
            Outputs.append(Output)
            RLaccs.append(RLacc)
            Hidden_State, Input, Output, RLacc = [], [], [], []
            Hidden_State = [0, 0, 0, math.cos(datas[i][25] * math.pi / 180),
                            math.sin(datas[i][25] * math.pi / 180), datas[i][24], datas[i][23] - last_height]
            Input.append(
                [datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19],
                 datas[i][20]]
            )
            q0, q1, q2, q3 = datas[i][16], datas[i][17], datas[i][18], datas[i][19]
            Matrix_R = np.array(
                [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
                 [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                 [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])
            Matrix_A = np.array([[datas[i][4]], [datas[i][5]], [datas[i][6]]])
            Matrix_A = np.dot(Matrix_R, Matrix_A).tolist()
            RLacc.append(
                [Matrix_A[0][0], Matrix_A[1][0], Matrix_A[2][0]]
            )
            last_time, lastE, lastL, last_height = datas[i][29], datas[i][21], datas[i][22], datas[i][23]

Hidden_States = np.array(Hidden_States)
Inputs = np.array(Inputs)
Outputs = np.array(Outputs)
RLaccs = np.array(RLaccs)

np.save(DIR + FILE_NAME + r'_Hidden_States.npy', Hidden_States)
np.save(DIR + FILE_NAME + r'_Inputs.npy', Inputs)
np.save(DIR + FILE_NAME + r'_Outputs.npy', Outputs)
np.save(DIR + FILE_NAME + r'_RLaccs.npy', RLaccs)
