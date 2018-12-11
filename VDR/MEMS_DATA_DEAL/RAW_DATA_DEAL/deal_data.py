from VDR.tools import *
import csv
import numpy as np
import random
import pandas as pd

DIR = r'D:\研究所\重点研究计划\data\MEMS_UBLOX'

MEMS_FILE = DIR + r'\mems\HL20180824021519_0IMUFIX.csv'
UBLOX_FILE = DIR + r'\ublox\HL20180824021519_0.csv'
TRUTH_FILE = DIR + r'\truth\truth.csv'

# 1.处理UBLOX信息（GGA）
csv_read = csv.reader(open(UBLOX_FILE, 'r', encoding='UTF-8'))
UBLOX = []
for row in csv_read:
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
    UBLOX.append([float(TIME), float(Longitude), float(Latitude), float(Height), float(HDOP), float(Mode)])
# UBLOX = np.array(UBLOX)
print('UBLOX信息处理完毕')
# UBLOX信息处理完毕

# 2.处理真值信息
csv_read = csv.reader(open(TRUTH_FILE, 'r', encoding='UTF-8'))
TRUTH = []
for row in csv_read:
    TRUTH.append([float(row[1]), float(row[3]), float(row[2]), float(row[4]),
                  float(row[6]), float(row[5]), float(row[7]) * -1,
                  float(row[8]), float(row[9]), float(row[10])])
# TRUTH = np.array(TRUTH)
print('TRUTH信息处理完毕')
# 真值信息处理完毕

# 3.处理MEMS信息
csv_read = csv.reader(open(MEMS_FILE, 'r', encoding='UTF-8'))
This_time = 0
MEMS = []
for row in csv_read:
    if This_time != float(row[1]):
        MEMS.append([[float(row[1]), float(row[5]), float(row[6]),
                      float(row[7]), float(row[2]), float(row[3]), float(row[4])]])
        This_time = float(row[1])
    else:
        MEMS[-1].append([float(row[1]), float(row[5]), float(row[6]), float(row[7]),
                         float(row[2]), float(row[3]), float(row[4])])
MEMS.pop(-1)
for i in MEMS:
    if len(i) != 50:
        i.append(i[random.randint(0, len(i) - 1)])
# MEMS = np.array(MEMS)
print('MEMS信息处理完毕')
# MEMS信息处理完毕

# 4.根据时间戳进行二次处理
UBLOX_START_TIME, UBLOX_STOP_TIME = UBLOX[0][0], UBLOX[-1][0]
TRUTH_START_TIME, TRUTH_STOP_TIME = TRUTH[0][0], TRUTH[-1][0]
MEMS_START_TIME, MEMS_STOP_TIME = MEMS[0][0][0], MEMS[-1][0][0]

START_TIME, STOP_TIME = max((UBLOX_START_TIME, TRUTH_START_TIME, MEMS_START_TIME)), \
                        min((UBLOX_STOP_TIME, TRUTH_STOP_TIME, MEMS_STOP_TIME))
while True:
    if UBLOX[0][0] < START_TIME:
        UBLOX.pop(0)
    elif UBLOX[-1][0] > STOP_TIME:
        UBLOX.pop(-1)
    else:
        break

while True:
    if TRUTH[0][0] < START_TIME:
        TRUTH.pop(0)
    elif TRUTH[-1][0] > STOP_TIME:
        TRUTH.pop(-1)
    else:
        break

while True:
    if MEMS[0][0][0] < START_TIME:
        MEMS.pop(0)
    elif MEMS[-1][0][0] >= STOP_TIME:
        MEMS.pop(-1)
    else:
        break

diff_pos = 0
for i in range(len(UBLOX)):
    while UBLOX[i][0] != TRUTH[i][0]:
        if diff_pos == 0:
            diff_pos = i
        UBLOX.insert(i, [TRUTH[i][0], 0, 0, 0, 100, 0])
        # TRUTH.remove(TRUTH[i])
        # MEMS.remove(MEMS[i])
# 二次处理完毕
print(UBLOX[diff_pos][0])

Inputs, States, Truths = [], [], []
for i in range(0, diff_pos - 1):
    Inputs.append(MEMS[i])
    States.append([UBLOX[i], UBLOX[i + 1]])
    Truths.append([TRUTH[i], TRUTH[i + 1]])

for i in range(diff_pos - 1, len(UBLOX) - 1):
    Inputs.append(MEMS[i])
    States.append([UBLOX[i], UBLOX[i + 1]])
    Truths.append([TRUTH[i], TRUTH[i + 1]])

Inputs = np.array(Inputs)
States = np.array(States)
Truths = np.array(Truths)

np.save(DIR + r'\Inputs.npy', Inputs)
np.save(DIR + r'\States.npy', States)
np.save(DIR + r'\Truths.npy', Truths)
