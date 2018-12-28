from VDR.tools import *
import csv

FILE = r'D:\研究所\重点研究计划\data\1015\20181015.dat'
csv_read = csv.reader(open(FILE, 'r'))
csv_write = csv.writer(open(r'D:\研究所\重点研究计划\data\1015\RTK.csv', 'w', newline=''))
for row in csv_read:
    if 'GGA' in row[0]:
        csv_write.writerow([float(row[1]), NMEA2LL(row[4]), NMEA2LL(row[2])])