from VDR.tools import *
import csv
out_file = csv.reader(open(r'D:\迅雷下载\RTKLIB_bin-rtklib_2.4.3\bin\out.csv', 'r'))
in_file = csv.writer(open(r'D:\迅雷下载\RTKLIB_bin-rtklib_2.4.3\bin\in.csv', 'w', newline=''))

datas = []
for row in out_file:
    if len(row) == 0:
        continue
    if 'GGA' in row[0]:
        # datas.append([float(NMEA2LL(row[4])), float(NMEA2LL(row[2]))])
        datas.append(row)

for i in datas:
    in_file.writerow(i)