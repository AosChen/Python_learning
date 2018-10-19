from cdf import CDF_Print_By_Array
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import preprocessing
import math
import random


# deal_main()

def getposition(v, a):
    return v * 0.01 + 0.5 * a * 0.01 * 0.01


DIR = r'D:\研究所\重点研究计划\data\1015\dealed_file'

Hidden_States = np.load(DIR + r'\Hidden_States.npy')
Inputs = np.load(DIR + r'\Inputs.npy')
Outputs = np.load(DIR + r'\Outputs.npy')
P_INS_Target = np.load(DIR + r'\P_INS_Target.npy')
P_INS = np.load(DIR + r'\P_INS.npy')
V_DEAL = np.load(DIR + r'\V_Deal.npy')

# Inputs = np.load('input.npy')
# Outputs = np.load('output.npy')

# 获取ngi每个时间戳的位置
NGI_FILE = DIR + r'\ngi.csv'
NGI_DATA = {}
with open(NGI_FILE, 'r') as ngi:
    ngi_reader = csv.reader(ngi)
    for row in ngi_reader:
        NGI_DATA[float(row[9])] = [float(row[1]), float(row[2])]

# 获取RTK每个时间戳的位置
RTK_FILE = DIR + r'\RTK_pos.csv'
RTK_DATA = {}
with open(RTK_FILE, 'r') as rtk:
    rtk_reader = csv.reader(rtk)
    for row in rtk_reader:
        RTK_DATA[float(row[0])] = [float(row[1]), float(row[2])]

# 总测试数据大小
data_size = Outputs.shape[0]

# 神经网络模型参数
INPUT_DIM = 16
OUTPUT_DIM = 5
INPUT_STEP = 100
BATCH_SIZE = 20
EPOCH_NUM = 100


# 模型定义
class INS_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(INS_NN, self).__init__()
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 16, batch_first=True)
        self.lstm3 = nn.LSTM(16, output_size, batch_first=True)

    def set_hidden(self, hidden_2):
        self.hidden2 = (hidden_2, torch.zeros(hidden_2.shape[0], hidden_2.shape[1], hidden_2.shape[2]))

    def forward(self, input):
        LSTM1_output, self.hidden1 = self.lstm1(input, None)
        LSTM2_output, self.hidden2 = self.lstm2(LSTM1_output, None)
        LSTM3_output, self.hidden3 = self.lstm3(LSTM2_output, None)
        return LSTM3_output[:, -1, :]


INS_NN_MODEL = INS_NN(INPUT_DIM, OUTPUT_DIM)
loss_function = nn.MSELoss()
optimizer = optim.Adam(INS_NN_MODEL.parameters(), lr=0.1)

Train_Size = int(Inputs.shape[0] * 0.8)
Test_Size = Inputs.shape[0] - Train_Size

# Inputs = np.array(Inputs)
# 数据集处理
Train_Inputs = torch.from_numpy(Inputs[:Train_Size]).float()
Train_Targets = torch.from_numpy(P_INS_Target[:Train_Size]).float()
# Train_Targets = torch.from_numpy(Outputs[:Train_Size]).float()

dataset = Data.TensorDataset(Train_Inputs, Train_Targets)
data_loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE
)

# for epoch in range(EPOCH_NUM):
#     for step, (batch_x, batch_y) in enumerate(data_loader):
#         input = batch_x
#         output = batch_y
#
#         INS_NN_MODEL.zero_grad()
#         result = INS_NN_MODEL(input)
#
#         loss = loss_function(result, output)
#         print('Epoch:', epoch, '| Step:', step, '| loss is ', str(loss.item()))
#         loss.backward()
#         optimizer.step()
#
# torch.save(INS_NN, DIR + r'\INS_NN.pkl')
INS_NN = torch.load(DIR + r'\INS_NN.pkl')

'''
以下为测试部分
模型的输入为：三轴线性加速度，四元数，气压计，三轴陀螺仪，一秒内的速度模值初值（整体除以100），一秒内的航向角初值（正余弦值），
            INS一秒内的东向和北向位移，共16维数据
模型的输出为：INS一秒内东向和北向位移的误差，一秒内的速度模值改变量，一秒内的航向角改变量（正余弦值）
测试部分：设定一个GPS的outrange部分（初始为5s），初始速度部分和航向角部分，根据前一秒的GPS数值进行设置，之后的则根据模型的
            预估进行设置。每一次计算获得一次INS的位移误差，根据该误差来修复INS的位移，并将修复好的位移输入给下一次模型的计算
            最终计算得出outrange部分的总位移，根据时间戳和RTK进行对比，获取到一次的误差。
'''
OUTRANGE_TIME = 5
error = []
for i in range(Train_Size, data_size - 4):
    px_ins, py_ins, px_rtk, py_rtk = 0, 0, 0, 0
    Vx_init, Vy_init = Hidden_States[i][6] * Hidden_States[i][4], Hidden_States[i][6] * Hidden_States[i][5]
    V_mod = math.sqrt(Vx_init * Vx_init + Vy_init * Vy_init)
    cos_yaw, sin_yaw = Hidden_States[i][4], Hidden_States[i][5]
    result = [0 for _ in range(OUTPUT_DIM)]
    for j in range(OUTRANGE_TIME):
        TEMP_INPUT = Inputs[i + j]
        px_ins_temp, py_ins_temp = 0, 0
        TIME_CUMP = float(Outputs[i + j][0])
        px_rtk += RTK_DATA[TIME_CUMP][0]
        py_rtk += RTK_DATA[TIME_CUMP][1]

        Vx_init = (V_mod + result[2] * 100) * (cos_yaw + result[3])
        Vy_init = (V_mod + result[2] * 100) * (sin_yaw + result[4])
        V_mod, cos_yaw, sin_yaw = math.sqrt(Vx_init * Vx_init + Vy_init * Vy_init), cos_yaw + result[3], sin_yaw + \
                                  result[4]
        for _ in range(5):
            TEMP_INPUT = np.delete(TEMP_INPUT, -1, axis=1)
        for temp_data in TEMP_INPUT:
            q0, q1, q2, q3 = temp_data[3], temp_data[4], temp_data[5], temp_data[6]
            Matrix_R = np.array(
                [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
                 [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                 [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
            )
            Matrix_A = temp_data[0:3]
            Matrix_A = np.dot(Matrix_R, Matrix_A).tolist()
            px_ins_temp += getposition(Vx_init, Matrix_A[0])
            py_ins_temp += getposition(Vy_init, Matrix_A[1])
        temp = np.transpose(
            np.array([V_mod / 100, cos_yaw, sin_yaw, px_ins_temp, py_ins_temp]).repeat(100).reshape(-1, 100))
        TEMP_INPUT = np.hstack([TEMP_INPUT, temp])
        result = INS_NN_MODEL(
            torch.from_numpy(
                TEMP_INPUT.reshape(-1, TEMP_INPUT.shape[0], TEMP_INPUT.shape[1])
            ).float()
        ).detach().numpy().flatten()
        px_ins += px_ins_temp + result[0]
        py_ins += py_ins_temp + result[1]
    error.append(math.sqrt((px_ins - px_rtk) ** 2 + (py_ins - py_rtk) ** 2))
    print('Step:', i+1-Train_Size, '| error is ', error[-1])
CDF_Print_By_Array([error], ['error'], 1, 1)
