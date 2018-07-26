#  本py文件的作用是绘制cdf图，其输入要求为csv文件，csv文件用','隔开，且每一行代表一条cdf曲线
import matplotlib.pyplot as plt

filename = r'D:\研究所\惯导安卓\data\0710\20180709195642_HUAWEI NXT-AL10_水平放置_副驾驶_error.csv'  # CSV文件路径

lines = []
with open(filename, 'r') as f:
    lines = f.read().split('\n')

dataSets = []

for line in lines:
    # print(line)
    try:
        dataSets.append(line.split(','))
    except:
        print("Error: Exception Happened... \nPlease Check Your Data Format... ")

temp = []
for set in dataSets:
    temp2 = []
    for item in set:
        if item != '':
            temp2.append(float(item))
    temp2.sort()
    temp.append(temp2)
dataSets = temp

for set in dataSets:

    plotDataset = [[], []]
    count = len(set)
    for i in range(count):
        plotDataset[0].append(float(set[i]))
        plotDataset[1].append((i + 1) / count)
    print(plotDataset)
    plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)

plt.show()
