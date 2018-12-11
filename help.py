import cdf
import pandas as pd

FILE = r'D:\研究所\重点研究计划\data\1011\20181011202759_HUAWEI NXT-AL10__100Hz - 副本error.csv'
FILE = open(FILE)
datas = pd.read_csv(FILE, names=['all_error', 'east_error', 'north_error',
                                 'all_error_rtk', 'east_error_rtk', 'north_error_rtk',
                                 'east_speed_error', 'north_speed_error'])
# Error = datas['error'].tolist()
# E_error = datas['E_error'].tolist()
# N_error = datas['N_error'].tolist()
print(datas)
labels = datas.columns.values.tolist()
Datas2Print = []
for label in labels:
    Datas2Print.append(datas[label].tolist())

cdf.figure()
cdf.CDF_One_Print(Datas2Print[0:4:3], labels[0:4:3], '10s GNSS outrage误差分布图')
cdf.figure()
cdf.CDF_One_Print(Datas2Print[1:5:3], labels[1:5:3], '10s GNSS outrage东向误差分布图')
cdf.figure()
cdf.CDF_One_Print(Datas2Print[2:6:3], labels[2:6:3], '10s GNSS outrage北向误差分布图')
cdf.figure()
cdf.CDF_One_Print(Datas2Print[6:8], labels[6:8], '10s GNSS outrage速度误差分布图')
cdf.show()
