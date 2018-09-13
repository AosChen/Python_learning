from VDR.Deal_Raw_Data import DIR
from VDR.Deal_Raw_Data import FILE_NAME
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class INS_Model(nn.Module):
    def __init__(self, hidden_dim, input_size):
        super(INS_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden = (torch.zeros(1, 1, hidden_dim),
                       torch.zeros(1, 1, hidden_dim))
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.gru = nn.GRU(input_size, hidden_dim)
        self.Liner1 = nn.Linear(hidden_dim, hidden_dim)

    def set_hidden(self, hidden_info):
        # self.hidden = (hidden_info.view(1, 1, self.hidden_dim),
        #                torch.zeros(1, 1, self.hidden_dim))
        self.hidden = hidden_info.view(1, 1, self.hidden_dim)

    def forward(self, input):
        # lstm_out, self.hidden = self.lstm(input.view(input.shape[0], 1, -1), self.hidden)
        gru_out, self.hidden = self.gru(input.view(input.shape[0], 1, -1), self.hidden)
        # result = lstm_out.view(input.shape[0], -1)
        result = self.Liner1(gru_out.view(input.shape[0], -1))
        result = F.tanh(result)
        return result[-1]


Hidden_States = np.load(DIR + FILE_NAME + r'_Hidden_States.npy')
Inputs = np.load(DIR + FILE_NAME + r'_Inputs.npy')
Outputs = np.load(DIR + FILE_NAME + r'_Outputs.npy')

HIDDEN_DIM = 7
INPUT_SIZE = 8
TIME_STEP = 100

INS_NN = INS_Model(HIDDEN_DIM, INPUT_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(INS_NN.parameters(), lr=0.1)

Data_Size = Hidden_States.shape[0]
Train_Size = Data_Size * 0.8
Test_Size = Data_Size - Train_Size

for i in range(int(Train_Size)):
    hidden_i = torch.from_numpy(Hidden_States[i]).float()
    input_i = torch.from_numpy(Inputs[i]).float()
    output_i = torch.from_numpy(Outputs[i, :HIDDEN_DIM]).float()

    INS_NN.zero_grad()
    INS_NN.set_hidden(hidden_i)
    result = INS_NN(input_i)
    loss = loss_function(result, output_i)
    print("The " + str(i) + "th itr: loss is " + str(loss.item()))
    print("The " + str(i) + "th itr: distance error is "
          + str(
        math.sqrt(pow(result[0].item() - output_i[0].item(), 2) + pow(result[1].item() - output_i[1].item(), 2))))
    print("======================================================")
    loss.backward()
    optimizer.step()

torch.save(INS_NN, DIR + FILE_NAME + r'_INS_NN.pkl')
INS_NN = torch.load(DIR + FILE_NAME + r'_INS_NN.pkl')

Test_Step = 5

positions = []
for i in range(int(Train_Size), Data_Size - Test_Step):
    hidden_i = torch.from_numpy(Hidden_States[i]).float()
    position = [0, 0]
    GNSS_position = [0, 0]
    for j in range(Test_Step):
        input_i = torch.from_numpy(Inputs[i + j]).float()
        output_i = torch.from_numpy(Outputs[i + j]).float()
        if j == 0:
            INS_NN.set_hidden(hidden_i)
        result = INS_NN(input_i)
        position[0] += result[0].item()
        position[1] += result[1].item()
        GNSS_position[0] += output_i[0].item()
        GNSS_position[1] += output_i[1].item()
        result[0] = 0.0
        result[1] = 0.0
        result[2] = 0.0
        INS_NN.set_hidden(result)
    distance1 = math.sqrt(position[0] * position[0] + position[1] * position[1])
    distance2 = math.sqrt(GNSS_position[0] * GNSS_position[0] + GNSS_position[1] * GNSS_position[1])
    print('步长为' + str(Test_Step) + 's，第' + str(i) + '组数据距离误差为：' + str(abs(distance1)) + 'm，当前GNSS时间为' + str(
        Outputs[i + 4, 7]))
    positions.append(distance1 - distance2)

error = open(DIR + FILE_NAME + r'_distance_test.csv', 'w')
for i in positions:
    error.write(str(abs(i)) + ',')
