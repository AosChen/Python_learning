from VDR.Deal_Raw_Data import DIR
from VDR.Deal_Raw_Data import FILE_NAME
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Hidden_States = np.load(DIR + FILE_NAME + r'_Hidden_States.npy')
Inputs = np.load(DIR + FILE_NAME + r'_Inputs.npy')
Outputs = np.load(DIR + FILE_NAME + r'_Outputs.npy')
RLaccs = np.load(DIR + FILE_NAME + r'_RLaccs.npy')

data_size = Hidden_States.shape[0]
Time_Step = 1 / Inputs.shape[1]

P_INSs = []

for i in range(data_size):
    px, py, pz = 0, 0, 0
    for j in range(Inputs.shape[1]):
        vx, vy, vz = Hidden_States[i][5] * Hidden_States[i][3], Hidden_States[i][5] * Hidden_States[i][4], \
                     Hidden_States[i][6]
        px += vx * Time_Step + 0.5 * RLaccs[i][j][0] * Time_Step * Time_Step
        py += vy * Time_Step + 0.5 * RLaccs[i][j][1] * Time_Step * Time_Step
        pz += vz * Time_Step + 0.5 * RLaccs[i][j][2] * Time_Step * Time_Step
        vx += RLaccs[i][j][0] * Time_Step
        vy += RLaccs[i][j][1] * Time_Step
        vz += RLaccs[i][j][2] * Time_Step
    P_INSs.append([px, py, pz])

P_INSs = np.array(P_INSs)
np.save(DIR + FILE_NAME + r'_P_INSs.npy', P_INSs)

INPUT_DIM = 8
OUTPUT_DIM = 3
INPUT_STEP = 100


class INS_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(INS_NN, self).__init__()
        self.Liner1 = nn.Linear(input_size, 128)
        self.rnn1 = nn.RNN(128, 64)
        self.rnn2 = nn.RNN(64, 16)
        self.Liner2 = nn.Linear(16, output_size)
        self.hidden_rnn1 = torch.zeros(1, 1, 64)
        self.hidden_rnn2 = torch.zeros(1, 1, 16)

    def init_hidden(self, ):
        self.hidden_rnn1 = torch.randn(1, 1, 64)
        self.hidden_rnn2 = torch.randn(1, 1, 16)

    def forward(self, input):
        Liner1_output = F.tanh(self.Liner1(input))
        Liner1_output = Liner1_output.view(Liner1_output.shape[0], 1, Liner1_output.shape[1])
        rnn1_output, self.hidden_rnn1 = self.rnn1(Liner1_output, self.hidden_rnn1)
        rnn1_output = F.tanh(rnn1_output)
        rnn2_output, self.hidden_rnn2 = self.rnn2(rnn1_output, self.hidden_rnn2)
        rnn2_output = F.tanh(rnn2_output)
        result = self.Liner2(rnn2_output)
        return result[-1]


INS_NN_MODEL = INS_NN(INPUT_DIM, OUTPUT_DIM)
loss_function = nn.MSELoss()
optimizer = optim.Adam(INS_NN_MODEL.parameters(), lr=0.1)

Train_Size = int(Inputs.shape[0] * 0.8)
Test_Size = Inputs.shape[0] - Train_Size

for i in range(Train_Size):
    input_i = torch.from_numpy(Inputs[i]).float()

    INS_NN_MODEL.zero_grad()
    INS_NN_MODEL.init_hidden()
    result = INS_NN_MODEL(input_i)