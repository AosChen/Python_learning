import torch
import torch.nn as nn
import torch.optim as optim
from math import *
import numpy as np


def invSqrt(x):
    return 1 / sqrt(x)


def getRMatrix(Q):
    Temp = np.mat([[Q[0] ** 2 + Q[1] ** 2 - Q[2] ** 2 - Q[3] ** 2, 2 * (Q[1] * Q[2] + Q[0] * Q[3]),
                    2 * (Q[1] * Q[3] - Q[0] * Q[2])],
                   [2 * (Q[1] * Q[2] - Q[0] * Q[3]), Q[0] ** 2 - Q[1] ** 2 + Q[2] ** 2 - Q[3] ** 2,
                    2 * (Q[2] * Q[3] + Q[0] * Q[1])],
                   [2 * (Q[1] * Q[3] + Q[0] * Q[2]), 2 * (Q[2] * Q[3] - Q[0] * Q[1]),
                    Q[0] ** 2 - Q[1] ** 2 - Q[2] ** 2 + Q[3] ** 2]])
    return Temp


class Mahony(nn.Module):
    def __init__(self):
        super(Mahony, self).__init__()
        self.line1 = nn.Linear(1, 1, bias=False)
        self.line2 = nn.Linear(1, 1, bias=False)

    def forward(self, input):
        integralFB = torch.tensor([0.0, 0.0, 0.0])
        q0, q1, q2, q3 = input[0][0], input[0][0], input[0][0], input[0][0]
        for i in range(len(input[1])):
            ax, ay, az = input[1][i][0], input[1][i][1], input[1][i][2]
            G = torch.tensor([input[1][i][3], input[1][i][4], input[1][i][5]])
            recipNorm = invSqrt(ax * ax + ay * ay + az * az)
            ax *= recipNorm
            ay *= recipNorm
            az *= recipNorm
            halfvx = q1 * q3 - q0 * q2
            halfvy = q0 * q1 + q2 * q3
            halfvz = q0 * q0 + q3 * q3 - 0.5

            halfe = torch.tensor([ay * halfvz - az * halfvy, az * halfvx - ax * halfvz, ax * halfvy - ay * halfvx])
            integralFB += halfe

            if self.line2.__getattribute__('_parameters')['weight'].data.item() < 0:
                integralFB = torch.tensor([0.0, 0.0, 0.0])

            line1_out_x = self.line1(torch.tensor([halfe[0]]))
            line1_out_y = self.line1(torch.tensor([halfe[1]]))
            line1_out_z = self.line1(torch.tensor([halfe[2]]))

            line2_out_x = self.line2(torch.tensor([integralFB[0] / 50]))
            line2_out_y = self.line2(torch.tensor([integralFB[1] / 50]))
            line2_out_z = self.line2(torch.tensor([integralFB[2] / 50]))

            G[0] += line1_out_x.item() + line2_out_x.item()
            G[1] += line1_out_y.item() + line2_out_y.item()
            G[2] += line1_out_z.item() + line2_out_z.item()

            qa, qb, qc = q0, q1, q2
            q0 = q0 - G[0] * qb - G[1] * qc - G[2] * q3
            q1 = q1 + qa * G[0] + qc * G[2] - q3 * G[1]
            q2 = q2 + qa * G[1] - qb * G[2] + q3 * G[0]
            q3 = q3 + qa * G[2] + qb * G[1] - qc * G[0]

            recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
            q0 *= recipNorm
            q1 *= recipNorm
            q2 *= recipNorm
            q3 *= recipNorm

            q0 = q0.item()
            q1 = q1.item()
            q2 = q2.item()
            q3 = q3.item()

        return torch.tensor([q0, q1, q2, q3])


DIR = r'D:\研究所\重点研究计划\data\MEMS_UBLOX'
Inputs = np.load(DIR + r'\Inputs.npy') * 50
States = np.load(DIR + r'\States.npy')
Truths = np.load(DIR + r'\Truths.npy')

Eular = np.array([0.0] * 3)
Q = [0.0] * 4

Train_Size = int(Truths.shape[0] * 0.8)

mahony = Mahony()
optimizer = optim.Adam(mahony.parameters(), lr=0.01)
loss_f = nn.MSELoss()

for i in range(Train_Size):
    Eular = Truths[i][0][7:]
    Eular[0] += 180  # 该部分的作用是调整正反面
    Eular = np.deg2rad(Eular)
    Q = [cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
         cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
         - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
         cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
         sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)]
    IMU = []
    for j in range(50):
        IMU.append(Inputs[i][j][1:])
    Input = [Q, IMU]

    Eular = Truths[i][1][7:]
    Eular[0] += 180  # 该部分的作用是调整正反面
    Eular = np.deg2rad(Eular)
    target_Q = torch.tensor([cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
                             + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
                             cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
                             - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
                             cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
                             + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
                             sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
                             - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)])
    Output = mahony(Input)
    optimizer.zero_grad()
    loss = loss_f(Output, target_Q)
    print("四元数误差：" + str(loss))
    print("欧拉角误差：")
    Cbn = getRMatrix(Output)
    Eular[0] = degrees(atan2(Cbn[2, 1], Cbn[2, 2]))
    Eular[1] = degrees(atan2(-Cbn[2, 0], sqrt(Cbn[2, 1] * Cbn[2, 1] + Cbn[2, 2] * Cbn[2, 2])))
    Eular[2] = -degrees(atan2(Cbn[1, 0], Cbn[0, 0]))
    if Eular[2] < 0:
        Eular[2] += 360
    print("真值欧拉角：" + str(Truths[i][1][7]) + "," + str(Truths[i][1][8]) + "," + str(Truths[i][1][9]))
    print("计算欧拉角：" + str(Eular[0]) + "," + str(Eular[1]) + "," + str(Eular[2]))
    print("------------------------------------------------------------------------------------------")
    loss.backward()
    optimizer.step()


Eular_error = []
for i in range(Train_Size, Truths.shape[0]):
    Eular = Truths[i][0][7:]
    Eular[0] += 180  # 该部分的作用是调整正反面
    Eular = np.deg2rad(Eular)
    Q = [cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
         cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
         - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
         cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
         sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)]
    IMU = []
    for j in range(50):
        IMU.append(Inputs[i][j][1:])
    Input = [Q, IMU]

    Eular = Truths[i][1][7:]
    Eular[0] += 180  # 该部分的作用是调整正反面
    Eular = np.deg2rad(Eular)
    target_Q = torch.tensor([cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
                             + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
                             cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
                             - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
                             cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
                             + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
                             sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
                             - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)])

    Output = mahony(Input)
    Cbn = getRMatrix(Output)
    Eular[0] = degrees(atan2(Cbn[2, 1], Cbn[2, 2]))
    Eular[1] = degrees(atan2(-Cbn[2, 0], sqrt(Cbn[2, 1] * Cbn[2, 1] + Cbn[2, 2] * Cbn[2, 2])))
    Eular[2] = -degrees(atan2(Cbn[1, 0], Cbn[0, 0]))
    if Eular[2] < 0:
        Eular[2] += 360
    print("真值欧拉角：" + str(Truths[i][1][7]) + "," + str(Truths[i][1][8]) + "," + str(Truths[i][1][9]))
    print("计算欧拉角：" + str(Eular[0]) + "," + str(Eular[1]) + "," + str(Eular[2]))
    roll_error = abs(Truths[i][1][7] - Eular[0])
    if roll_error > 180:
        roll_error = 360 - roll_error
    pitch_error = abs(Truths[i][1][8] - Eular[1])
    if pitch_error > 180:
        pitch_error = 360 - pitch_error
    yaw_error = abs(Truths[i][1][9] - Eular[2])
    if yaw_error > 180:
        yaw_error = 360 - yaw_error
    Eular_error.append([roll_error, pitch_error, yaw_error])
    print("------------------------------------------------------------------------------------------")
