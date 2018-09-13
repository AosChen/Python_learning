import csv

DIR = r'C:\Users\AosChen\Desktop\混合定位引擎二期合作验收20180726\各子模块算法实现、源码和测试中间数据\4.VDR\data\100Hz'
FILE = DIR + r'\20180725190248_HUAWEI NXT-AL10_水平放置_副驾驶_.csv'

csv_reader = csv.reader(open(FILE, 'r'))

attr = []
datas = []
index = 0

for row in csv_reader:
    if index == 0:
        print(row)
        attr = row
        index += 1
    else:
        datas.append(list(map(float, row)))

this_time = 0
new_datas = []
for i in range(len(datas))[::10]:
    if datas[i][29] != this_time:
        if this_time != 0:
            while datas[i][29] != this_time:
                i -= 1
            i += 1  # 找到第一个新时刻起始点
    this_time = datas[i][29]
    new_datas.append(datas[i])
print(new_datas)

csv_writer = csv.writer(open(DIR + r'\new_file.csv', 'w', newline=""))
for i in new_datas:
    csv_writer.writerow(i)
