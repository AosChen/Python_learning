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
        self.hidden2result = nn.Linear(hidden_dim, hidden_dim)

    def set_hidden(self, hidden_info):
        self.hidden = (hidden_info.view(1, 1, self.hidden_dim),
                       torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(input.shape[0], 1, -1), self.hidden)
        result = self.hidden2result(lstm_out.view(input.shape[0], -1))
        result = F.relu(result)
        return result[-1]


Hidden_States = np.load('Hidden_States.npy')
Inputs = np.load('Inputs.npy')
Outputs = np.load('Outputs.npy')

HIDDEN_DIM = 4
INPUT_SIZE = 19
TIME_STEP = 5

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
    print("The " + str(i) + "th itr: loss is " + str(loss[0]))
    loss.backward()
    optimizer.step()