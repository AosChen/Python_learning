import matplotlib.pyplot as plt


def CDF_Print_By_Array(datas, titles, rows, columns):
    length = len(titles)
    for i in range(length):
        plt.subplot(rows, columns, i + 1)
        plt.title(titles[i])
        plt.grid(True)
        plotDataset = [[], []]
        datas[i].sort()
        count = len(datas[i])
        for j in range(count):
            plotDataset[0].append(float(datas[i][j]))
            plotDataset[1].append((j + 1) / count)
        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
    plt.show()


def CDF_One_Print(datas, labels, title):
    length = len(labels)
    for i in range(length):
        plt.grid(True)
        plotDataset = [[], []]
        datas[i].sort()
        count = len(datas[i])
        for j in range(count):
            plotDataset[0].append(float(datas[i][j]))
            plotDataset[1].append((j + 1) / count)
        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
    plt.legend(labels)
    plt.title(title, fontproperties="SimHei")
    # plt.show()


def figure():
    plt.figure()


def show():
    plt.show()
