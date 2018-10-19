import time
import csv
import numpy as np

FILE = r"D:\研究所\重点研究计划\data\1015\20181015155859_HUAWEI NXT-AL10__100Hz.csv"
with open(FILE, 'r') as ngi_file:
    ngi_csv = csv.reader(ngi_file)
    index = 0
    for row in ngi_csv:
        if index == 0:
            index += 1
            continue
        row[0] = float(row[0])
        temp = row[0] / 1000
        # print(temp)
        MS = temp - int(temp)
        temp = time.localtime(temp)
        temp = time.strftime("%H:%M:%S", temp)
        print(temp, ' ', MS, ' ', row[-1])
