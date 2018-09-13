#  本py文件的作用是绘制cdf图，其输入要求为csv文件，csv文件用','隔开，且每一行代表一条cdf曲线
import matplotlib.pyplot as plt

# Title = ['20-30km/h Straight cdf', '40-60km/h Straight cdf', 'Turn cdf', 'Sprial cdf']
Title = ['5s_GPS_blocked_cdf', '40-60km/h Straight cdf', 'Turn cdf', 'Sprial cdf']
# filename = r'D:\研究所\惯导安卓\data\0710\直角 my.csv'  # CSV文件路径
# DIR = r'D:\研究所\惯导安卓\data\0710'
# filenames = [DIR + r'\20-30.csv', DIR + r'\40-60.csv', DIR + r'\转弯.csv', DIR + r'\Sprial.csv']
file_lines = []
filenames = [r'D:\研究所\重点研究计划\data\0824\distance_test.csv']

for filename in filenames:
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        file_lines.append(lines)

dataSets = []
for lines in file_lines:
    dataSet = []
    for line in lines:
        # print(line)
        try:
            dataSet.append(line.split(','))
        except:
            print("Error: Exception Happened... \nPlease Check Your Data Format... ")
    dataSets.append(dataSet)

temps = []
for dataSet in dataSets:
    temp = []
    for set in dataSet:
        temp2 = []
        for item in set:
            if item != '':
                temp2.append(float(item))
        temp2.sort()
        temp.append(temp2)
    temps.append(temp)
dataSets = temps

j = 0
for dataSet in dataSets:
    # plt.subplot(220+j+1)
    plt.title(Title[j])
    plt.grid(True)
    for set in dataSet:
        plotDataset = [[], []]
        count = len(set)
        for i in range(count):
            plotDataset[0].append(float(set[i]))
            plotDataset[1].append((i + 1) / count)
        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
    j += 1
# plt.grid(color='b', linewidth='0.3', linestyle='--')
plt.show()
