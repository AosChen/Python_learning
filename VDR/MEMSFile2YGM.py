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


MEMSFile = csv.reader(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\mems\HL20180824021519_0IMUFIX.csv', 'r'))
UBLOXFile = csv.reader(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\ublox\HL20180824021519_0.csv', 'r'))
TruthFile = csv.reader(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\truth\truth.csv', 'r'))

MEMS = {}
for row in MEMSFile:
    row = list(map(float, row))
    key = row[1]
    if key in MEMS.keys():
        MEMS[key].append([row[2] * 50 * rad2deg, row[3] * 50 * rad2deg, row[4] * 50 * rad2deg,
                          row[5] * 50, row[6] * 50, row[7] * 50])
    else:
        MEMS[key] = [[row[2], row[3], row[4], row[5], row[6], row[7]]]


OutputFile = csv.writer(open(r'D:\SELF_DATA\YGM_in.csv', 'w', newline=''))
