import csv
import math
import numpy as np
import random

rad2deg = 180 / math.pi

def p_range(val, min, max):
    if val > max:
        val = max
    elif val < min:
        val = min
    return val


def atan2EX(y, x):
    if sign(y) == 0 and sign(x) == 0:
        return 0.0
    else:
        return math.atan2(y, x)


def sign(x):
    if x < 2.220446049e-16 * -1:
        return -1
    elif x > 2.220446049e-16:
        return 1
    else:
        return 0


def q2a(q0, q1, q2, q3):
    q11, q12, q13, q14 = q0 * q0, q0 * q1, q0 * q2, q0 * q3
    q22, q23, q24 = q1 * q1, q1 * q2, q1 * q3
    q33, q34 = q2 * q2, q2 * q3
    q44 = q3 * q3

    return [math.degrees(math.asin(p_range(2 * (q34 + q12), -1, 1))),
            math.degrees(atan2EX(-2 * (q24 - q13), q11 - q22 - q33 + q44)),
            math.degrees(atan2EX(-2 * (q23 - q14), q11 - q22 + q33 - q44))]


InputFile = csv.reader(open(r'D:\研究所\重点研究计划\data\1011\20181011202759_HUAWEI NXT-AL10__100Hz - 副本.csv', 'r'))
OutputFile = csv.writer(open(r'D:\SELF_DATA\YGM_in.csv', 'w', newline=''))

datas = []
for row in InputFile:
    datas.append(row)
datas.pop(0)

last_UTC = 0
data_set = []
data_sets = []
for data in datas:
    data = list(map(float, data))
    if last_UTC != data[-1]:
        if len(data_sets) >= 2:
            while len(data_set) > 100:
                data_set.pop(random.randint(0, len(data_set) - 1))
            while len(data_set) < 100:
                data_set.append(data_set[random.randint(0, len(data_set) - 1)])
        data_sets.append(data_set)
        data_set = []
        last_UTC = data[-1]
        data_set.append(data)
    else:
        data_set.append(data)

# print(data_sets)
data_sets.pop(0)
for i in range(1, len(data_sets)):
    for j in range(100):
        Time = [i*j/100]
        Gyo = [data_sets[i][j][9] * rad2deg, data_sets[i][j][10] * rad2deg, data_sets[i][j][11] * rad2deg]
        Acc = [data_sets[i][j][0], data_sets[i][j][1], data_sets[i][j][2]]
        Eular = q2a(data_sets[i][j][15], data_sets[i][j][16], data_sets[i][j][17], data_sets[i][j][18])
        Nav_info = [0] * 6
        if j == 0:
            GPSVu = data_sets[i][j][-7] - data_sets[i - 1][j][-7]
            Yaw = math.radians(data_sets[i][j][-5])
            if Yaw > 180:
                write_yaw = 360 - Yaw
            else:
                write_yaw = -Yaw

            Gps_info = [math.sin(Yaw) * data_sets[i][j][-6], math.cos(Yaw) * data_sets[i][j][-6], GPSVu,
                        data_sets[i][j][-8], data_sets[i][j][-9], data_sets[i][j][-7],
                        write_yaw, data_sets[i][j][-2], data_sets[i][j][-4]]
        else:
            Gps_info = [0] * 9
        Mag = [data_sets[i][j][12] * 10, data_sets[i][j][13] * 10, data_sets[i][j][14] * 10]
        Baro = [44300 * (1 - math.pow(data_sets[i][j][19] / 10 / 101.325, 1 / 5.5256))]
        data2write = Time + Gyo + Acc + Eular + Nav_info + Gps_info + Mag + Baro
        OutputFile.writerow(data2write)