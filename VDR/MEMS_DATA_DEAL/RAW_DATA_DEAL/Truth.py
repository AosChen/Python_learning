import time
import csv


def GPSTIME2UTCTIME(weeks, times):
    weeksInt = -1
    weekTONow = 0.0

    weeksInt = int(weeks) - 1
    weekTONow = float(times)

    ms = weeksInt * 7 * (24 * 60 * 60) + time.mktime(time.strptime('1980-01-06', '%Y-%m-%d')) + weekTONow

    TIME = time.strftime('%Y%m%d,%H%M%S', time.localtime(ms)).split(',')
    return [TIME[0], TIME[1]]


truth_file = open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\truth\truth.nav', 'r')
in_file = csv.writer(open(r'D:\研究所\重点研究计划\data\MEMS_UBLOX\truth\truth.csv', 'w', newline=''))

for row in truth_file:
    row = row.split(' ')
    while '' in row:
        row.remove('')
    row.remove('\n')
    TIME = GPSTIME2UTCTIME(row[0], row[1])
    row.pop(0)
    row.pop(0)
    row.insert(0, TIME[0])
    row.insert(1, TIME[1])
    in_file.writerow(row)
