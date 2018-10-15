from VDR.Deal_Raw_Data import DIR
from VDR.Deal_Raw_Data import FILE_NAME
from VDR.Deal_Raw_Data import main as deal_main
from cdf import CDF_Print_By_Array
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import math

# deal_main()

Hidden_States = np.load(DIR + FILE_NAME + r'_Hidden_States.npy')
Inputs = np.load(DIR + FILE_NAME + r'_Inputs.npy')
Outputs = np.load(DIR + FILE_NAME + r'_Outputs.npy')
RLaccs = np.load(DIR + FILE_NAME + r'_RLaccs.npy')

data_size = Hidden_States.shape[0]
Time_Step = 1 / Inputs.shape[1]

P_INSs = []

for i in range(data_size):
    px, py, pz = 0, 0, 0
    vx, vy, vz = Hidden_States[i][5] * Hidden_States[i][3], Hidden_States[i][5] * Hidden_States[i][4], Hidden_States[i][
        6]
    for j in range(Inputs.shape[1]):
        px += vx * Time_Step + 0.5 * RLaccs[i][j][0] * Time_Step * Time_Step
        py += vy * Time_Step + 0.5 * RLaccs[i][j][1] * Time_Step * Time_Step
        pz += vz * Time_Step + 0.5 * RLaccs[i][j][2] * Time_Step * Time_Step
        vx += RLaccs[i][j][0] * Time_Step
        vy += RLaccs[i][j][1] * Time_Step
        vz += RLaccs[i][j][2] * Time_Step
    P_INSs.append([px, py, pz])

P_INSs = np.array(P_INSs)
np.save(DIR + FILE_NAME + r'_P_INSs.npy', P_INSs)

INPUT_DIM = 20
HIDDEN_DIM = 4
OUTPUT_DIM = 3
INPUT_STEP = 100
BATCH_SIZE = 50
EPOCH_NUM = 100


class INS_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(INS_NN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, output_size, batch_first=True)
        self.lstm3 = nn.LSTM(output_size, output_size, batch_first=True)

    def set_hidden(self, hidden_init):
        self.hidden1 = (hidden_init, torch.zeros(hidden_init.shape[0], hidden_init.shape[1], hidden_init.shape[2]))
        # self.hidden2 = (torch.randn(1, 1, self.output_size), torch.randn(1, 1, self.output_size))
        # self.hidden3 = (torch.randn(1, 1, self.output_size), torch.randn(1, 1, self.output_size))

    def forward(self, input):
        LSTM1_output, self.hidden1 = self.lstm1(input, self.hidden1)
        # LSTM1_output.view(LSTM1_output.shape[0], 1, -1)
        LSTM2_output, self.hidden2 = self.lstm2(LSTM1_output, None)
        # LSTM2_output.view(LSTM2_output.shape[0], 1, -1)
        LSTM3_output, self.hidden3 = self.lstm3(LSTM2_output, None)
        return LSTM3_output[:, -1, :]


INS_NN_MODEL = INS_NN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_function = nn.MSELoss()
optimizer = optim.Adam(INS_NN_MODEL.parameters(), lr=0.1)

Train_Size = int(Inputs.shape[0] * 0.8)
Test_Size = Inputs.shape[0] - Train_Size

Inputs = torch.from_numpy(Inputs).float()
Targets = torch.from_numpy(np.hstack((Hidden_States, Outputs, P_INSs))).float()

dataset = Data.TensorDataset(Inputs, Targets)
data_loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE
)

for epoch in range(EPOCH_NUM):
    for step, (batch_x, batch_y) in enumerate(data_loader):
        input = batch_x
        hidden = batch_y[:, 3:7].view(-1, batch_y.shape[0], 4)
        output = (batch_y[:, 7:10] - batch_y[:, -3:])

        INS_NN_MODEL.zero_grad()
        INS_NN_MODEL.set_hidden(hidden)
        result = INS_NN_MODEL(input)

        loss = loss_function(result, output)
        print('Epoch:', epoch, '| Step:', step, '| loss is ', str(loss.item()))
        loss.backward()
        optimizer.step()

torch.save(INS_NN, DIR + FILE_NAME + r'_INS_NN.pkl')
INS_NN = torch.load(DIR + FILE_NAME + r'_INS_NN.pkl')

Test_Step = 5
Test_errors = []
draw_datas = []
for i in range(int(Train_Size), data_size - Test_Step):
    px_ins, py_ins, pz_ins = 0, 0, 0
    px_gps, py_gps, pz_gps = 0, 0, 0
    vx_init, vy_init, vz_init = Hidden_States[i][5] * Hidden_States[i][3], Hidden_States[i][5] * Hidden_States[i][4], \
                                Hidden_States[i][6]
    hidden_i = torch.from_numpy(Hidden_States[i][3:]).float().view(1, 1, 4)
    INS_NN_MODEL.set_hidden(hidden_i)
    for j in range(Test_Step):
        px_gps += Outputs[i + j][0]
        py_gps += Outputs[i + j][1]
        pz_gps += Outputs[i + j][2]
        for k in range(Inputs.shape[1]):
            px_ins += vx_init * Time_Step + 0.5 * RLaccs[i][k][0] * Time_Step * Time_Step
            py_ins += vy_init * Time_Step + 0.5 * RLaccs[i][k][1] * Time_Step * Time_Step
            pz_ins += vz_init * Time_Step + 0.5 * RLaccs[i][k][2] * Time_Step * Time_Step
            vx_init += RLaccs[i][k][0] * Time_Step
            vy_init += RLaccs[i][k][1] * Time_Step
            vz_init += RLaccs[i][k][2] * Time_Step
        INS_error = INS_NN_MODEL(Inputs[i + j].view(-1, INPUT_STEP, INPUT_DIM))
        px_ins += INS_error[0][0].item()
        py_ins += INS_error[0][1].item()
        pz_ins += INS_error[0][2].item()

    distance_error = math.sqrt(math.pow(px_ins - px_gps, 2) + math.pow(py_ins - py_gps, 2))
    Test_errors.append(distance_error)

draw_datas.append(Test_errors)
CDF_Print_By_Array(draw_datas, [FILE_NAME[1:15] + '_error_cdf'], 1, 1)
