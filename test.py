import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acc_file = r'D:\研究所\重点研究计划\data\0927\20180926204123\20180926204123_HUAWEI NXT-AL10__100Hz_acc.csv'
gyo_file = r'D:\研究所\重点研究计划\data\0927\20180926204123\20180926204123_HUAWEI NXT-AL10__100Hz_gyo.csv'

acc_file = open(acc_file)
gyo_file = open(gyo_file)

acc = pd.read_csv(acc_file, names=list('tabc'))
gyo = pd.read_csv(gyo_file, names=list('tabc'))

t_std = np.array(list(map(float, acc[list('t')].values)))
t_self = np.array(list(map(float, gyo[list('t')].values)))

g_i = []
for c in 'abc':
    temp = np.interp(np.array(list(map(float, acc[list('t')].values))),
                     np.array(list(map(float, gyo[list('t')].values))),
                     np.array(list(map(float, gyo[list(c)].values))))
    g_i.append(temp)

plt.figure()
plt.plot(t_self, np.array(list(map(float, gyo[list('a')].values))), c='r')
plt.plot(t_std, g_i[0], c='g')

plt.figure()
plt.plot(t_self, np.array(list(map(float, gyo[list('b')].values))), c='r')
plt.plot(t_std, g_i[1], c='g')

plt.figure()
plt.plot(t_self, np.array(list(map(float, gyo[list('c')].values))), c='r')
plt.plot(t_std, g_i[2], c='g')

plt.show()