import numpy as np
import csv

# csv_reader = csv.reader(open(r"D:\研究所\惯导安卓\data\0710\20180709195642_HUAWEI NXT-AL10_水平放置_副驾驶_.csv"))
# index = 0
#
# attr = []
# datas = []
# for row in csv_reader:
#     if index == 0:
#         print(row)
#         attr = row
#         index += 1
#     else:
#         datas.append(list(map(float, row)))
# datas = np.array(datas)
# np.save('datas_0710.npy', datas)

datas = np.load('datas_0710.npy')
print(datas)