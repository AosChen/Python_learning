from VDR.MEMS_DATA_DEAL.RAW_DATA_DEAL.Truth import GPSTIME2UTCTIME
import csv
import struct

MEMS_FILE = r'D:\研究所\重点研究计划\data\MEMS_UBLOX\mems\HL20180824021519_0IMUFIX.bin'
RESULT_FILE = r'D:\研究所\重点研究计划\data\MEMS_UBLOX\mems\HL20180824021519_0IMUFIX.csv'

csvwriter = csv.writer(open(RESULT_FILE, 'w', newline=''))
file_reader = open(MEMS_FILE, 'rb')
co = 0
datas = []
while True:
    data = file_reader.read(8)
    if not data:
        break
    co += 1
    datas.append(struct.unpack('d', data)[0])
    if co == 7:
        TIME = GPSTIME2UTCTIME('2016', datas[0])
        datas.pop(0)
        datas.insert(0, TIME[0])
        datas.insert(1, TIME[1])
        csvwriter.writerow(datas)
        datas = []
        co = 0
file_reader.close()
