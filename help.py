import cdf
import pandas as pd

FILE = r'D:\研究所\重点研究计划\data\1015\change_file.csv'
FILE = open(FILE)
datas = pd.read_csv(FILE, names=['before KF', 'after KF'])
# Error = datas['error'].tolist()
# E_error = datas['E_error'].tolist()
# N_error = datas['N_error'].tolist()
print(datas)
labels = datas.columns.values.tolist()
Datas2Print = []
for label in labels:
    Datas2Print.append(datas[label].tolist())

cdf.figure()
cdf.CDF_One_Print(Datas2Print, labels, '1s内惯导推算与RTK真值的误差')
cdf.show()
