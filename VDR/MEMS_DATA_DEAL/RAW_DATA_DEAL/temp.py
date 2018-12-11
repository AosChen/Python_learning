from VDR.MEMS_DATA_DEAL.RAW_DATA_DEAL.Truth import GPSTIME2UTCTIME
from VDR.tools import *
import csv

file = csv.reader(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\ublox\HL20180824021519_0.csv', 'r'))
file2 = csv.reader(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\ublox.csv', 'r'))
error = csv.writer(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\other_ublox_error.csv', 'w', newline=''))

UBLOX1 = {}
UBLOX2 = {}
for row in file:
    UBLOX1[float(row[1])] = [NMEA2LL(row[4]), NMEA2LL(row[2])]
    error.writerow([float(row[1]), NMEA2LL(row[4]), NMEA2LL(row[2])])

for row in file2:
    UBLOX2[float(row[0])] = [float(row[2]), float(row[1])]

# for key in UBLOX2.keys():
#     if key not in UBLOX1.keys():
#         continue
#     error.writerow([key, distance_by_LoLa(UBLOX1[key][0], UBLOX1[key][1], UBLOX2[key][0], UBLOX2[key][1])])
