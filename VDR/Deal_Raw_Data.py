import numpy as np
import csv
import math

FILE_NAME = r"D:\研究所\重点研究计划\data\20180820085418_HUAWEI NXT-AL10__100Hz.csv"


def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = La1 * np.pi / 180.0
    radLa2 = La2 * np.pi / 180.0
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * math.asin(math.sqrt(
        math.pow(math.sin(deltaLa / 2), 2) + math.cos(radLa1) * math.cos(radLa2) * math.pow(math.sin(deltaLo / 2),
                                                                                            2))) * Er


csv_reader = csv.reader(open(FILE_NAME, 'r'))
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
np.save('datas_0710.npy', datas)

datas = np.load('datas_0710.npy')
print(datas)

datas_size = datas.shape[0]
datas_step = 100

Hidden_States, Inputs, Outputs = [], [], []
Hidden_State, Input, Output = [], [], []
last_height = datas[0][23] # 上一时刻的高程信息，用来粗略计算高程的速度
last_time = datas[0][29] # 上一时刻的NMEA的UTC时间，用来判断数据是否发生变化
i = 1 # 记录当前处理的位置坐标
size = 0 # 记录一个周期内已经处理了的数据个数
while True:
    this_time = datas[i][29]
    if this_time == last_time:
        Input.append([datas[i][4], datas[i][5], datas[i][6], datas[i][16], datas[i][17], datas[i][18], datas[i][19], datas[i][20]])
        i += 1
        size += 1
        continue
    else:
        Hidden_State = [0, 0, 0, ]
        if size < datas_step:
            while size < datas_step:
                Input.append(Input[-1])
                size += 1
        Output = []


    pass

Hidden_States = np.array(Hidden_States)
Inputs = np.array(Inputs)
Outputs = np.array(Outputs)

np.save('Hidden_States.npy', Hidden_States)
np.save('Inputs.npy', Inputs)
np.save('Outputs.npy', Outputs)
