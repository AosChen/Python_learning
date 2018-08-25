from VDR.Deal_Raw_Data import DIR
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class INS_Model(nn.Module):
    def __init__(self, hidden_dim, input_size):
        super(INS_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden = (torch.zeros(1, 1, hidden_dim),
                       torch.zeros(1, 1, hidden_dim))
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.Liner1 = nn.Linear(hidden_dim, hidden_dim)

    def set_hidden(self, hidden_info):
        self.hidden = (hidden_info.view(1, 1, self.hidden_dim),
                       torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(input.shape[0], 1, -1), self.hidden)
        result = self.Liner1(lstm_out.view(input.shape[0], -1))
        # result = F.relu(result)
        return result[-1]


Hidden_States = np.load(DIR + r'\Hidden_States.npy')
Inputs = np.load(DIR + r'\Inputs.npy')
Outputs = np.load(DIR + r'\Outputs.npy')

HIDDEN_DIM = 7
INPUT_SIZE = 8
TIME_STEP = 100

INS_NN = INS_Model(HIDDEN_DIM, INPUT_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(INS_NN.parameters(), lr=0.1)

Data_Size = Hidden_States.shape[0]

for i in range(Data_Size):
    hidden_i = torch.from_numpy(Hidden_States[i]).float()
    input_i = torch.from_numpy(Inputs[i]).float()
    output_i = torch.from_numpy(Outputs[i]).float()

    INS_NN.zero_grad()
    INS_NN.set_hidden(hidden_i)
    result = INS_NN(input_i)
    loss = loss_function(result, output_i)
    print("The " + str(i) + "th itr: loss is " + str(loss.item()))
    print("The " + str(i) + "th itr: distance error is "
          + str(math.sqrt(pow(result[0].item() - output_i[0].item(), 2) + pow(result[1].item() - output_i[1].item(), 2))))
    print("======================================================")
    loss.backward()
    optimizer.step()

torch.save(INS_NN, DIR + r'\INS_NN.pkl')
