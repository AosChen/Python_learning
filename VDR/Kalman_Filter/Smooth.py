class Smooth(object):
    # def __init__(self):
    #     self.index = 0
    #     self.max_size = 12
    #     self.smoothRes = []

    def __init__(self, max_size):
        self.index = 0
        self.max_size = max_size
        self.smoothRes = []

    def getSmoothResult(self, data):
        if len(self.smoothRes) < self.max_size:
            self.smoothRes.append(data)
        else:
            self.smoothRes[self.index] = data

        self.index = (self.index + 1) % self.max_size
        temp = [0] * len(data)
        for i in range(len(data)):
            for j in range(len(self.smoothRes)):
                temp[i] += self.smoothRes[j][i]
            temp[i] /= len(self.smoothRes)
        return temp
