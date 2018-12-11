import numpy as np
from math import *
from VDR.MEMS_DATA_DEAL.tools.MahonyAHRS import *
from VDR.MEMS_DATA_DEAL.tools.KF import *
from VDR.MEMS_DATA_DEAL.tools.Smooth import *
from VDR.tools import distance_by_LoLa
import csv


def getRMatrix(Q):
    Temp = np.mat([[Q[0] ** 2 + Q[1] ** 2 - Q[2] ** 2 - Q[3] ** 2, 2 * (Q[1] * Q[2] + Q[0] * Q[3]),
                    2 * (Q[1] * Q[3] - Q[0] * Q[2])],
                   [2 * (Q[1] * Q[2] - Q[0] * Q[3]), Q[0] ** 2 - Q[1] ** 2 + Q[2] ** 2 - Q[3] ** 2,
                    2 * (Q[2] * Q[3] + Q[0] * Q[1])],
                   [2 * (Q[1] * Q[3] + Q[0] * Q[2]), 2 * (Q[2] * Q[3] - Q[0] * Q[1]),
                    Q[0] ** 2 - Q[1] ** 2 - Q[2] ** 2 + Q[3] ** 2]])
    return Temp


def updateQ(P, V, G, Cbn, Q):
    Q = np.array(Q)
    Rm = Re * (1 - 2 * earthf + 3 * earthf * (sin(P[1][0]) ** 2))
    Rn = Re * (1 + earthf * (sin(P[1][0]) ** 2))

    w_n_ie = np.array([[0], [wie * cos(P[1][0])], [wie * sin(P[1][0])]])
    w_n_en = np.array([[-V[1][0] / Rm], [V[0][0] / Rn], [V[0][0] / Rn * tan(P[1][0])]])
    w_b_nb = G - Cbn.T * (w_n_ie + w_n_en)

    K1 = 0.5 * Q * w_b_nb[0, 0]
    Qk_1_2 = Q + 0.02 * K1
    K2 = 0.5 * Qk_1_2 * w_b_nb[1, 0]
    Qk_1_2_ = Q + 0.02 * K2
    K3 = 0.5 * Qk_1_2_ * w_b_nb[1, 0]
    Qk_ = Q + 2 * 0.05 * K3
    K4 = 0.5 * Qk_ * w_b_nb[2, 0]
    K = 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return Q + 2 * 0.02 * K



accCalStc = np.array([[0.0], [0.0], [0.0]])
gyoCalStc = np.array([[0.0], [0.0], [0.0]])

DIR = r'D:\研究所\重点研究计划\data\MEMS_UBLOX'
sampleFreq = 50
Re = 6378137
earthf = 1 / 298.257
wie = 0.000072921151467

Inputs = np.load(DIR + r'\Inputs.npy') * 50
States = np.load(DIR + r'\States.npy')
Truths = np.load(DIR + r'\Truths.npy')

# 初始定姿的数据，约10s
Time_Static = 10

Eular = np.array([0.0] * 3)
Velocity = np.array([0.0] * 3)
Position = np.array([0.0] * 3)
Q = [0.0] * 4

# 初始姿态的速度和位置
Velocity += Truths[Time_Static - 1][0][4:7]  # 东，北，地
Position += States[Time_Static - 1][0][1:4]  # 经度，纬度，高度

# 初始姿态确定
for i in range(Time_Static):
    Eular += Truths[i][1][7:]
Eular /= Time_Static
Eular = np.deg2rad(Eular)
Q = [cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
     + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
     cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
     - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
     cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
     + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
     sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
     - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)]
# 初始姿态确定完毕

# 力学公式计算INS，处理MEMS数据计算
ma = MahonyAHRS(sampleFreq)
KF = KalmanFilter()

V_INS = Velocity.reshape(3, 1)
P_INS = Position.reshape(3, 1)
# Eular = Eular.reshape(3, 1)
P_INS[0][0] = np.deg2rad(P_INS[0][0])
P_INS[1][0] = np.deg2rad(P_INS[1][0])
Cbn = getRMatrix(Q)
ma.setQ(Q)
doKF = True

Eular_error = []
csv_writer = csv.writer(open('Mahony_error.csv', 'w', newline=''))

smooth = Smooth(15)

for i in range(Time_Static, Inputs.shape[0]):
    # 更新重力等信息
    g = 978.03267714 * (1 + 0.00193185138639 * (sin(P_INS[1][0]) ** 2)) \
        / sqrt(1 - 0.00669437999013 * (sin(P_INS[1][0]) ** 2)) / 100
    g = np.array([[0], [0], [g]])
    Rm = Re * (1 - 2 * earthf + 3 * earthf * (sin(P_INS[1][0]) ** 2))
    Rn = Re * (1 + earthf * (sin(P_INS[1][0]) ** 2))
    Eular = Truths[i][0][7:]
    Eular = np.deg2rad(Eular)
    Q = [cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
         cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
         - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
         cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
         + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
         sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
         - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)]
    ma.setQ(Q)
    Cbn = getRMatrix(Q)

    # for j in range(Inputs.shape[1]):
    #     ax, ay, az = Inputs[i][j][2] - accCalStc[2][0], Inputs[i][j][2] - accCalStc[1][0], Inputs[i][j][3] - \
    #                  accCalStc[2][0]
    #     gx, gy, gz = Inputs[i][j][4] - gyoCalStc[0][0], Inputs[i][j][5] - gyoCalStc[1][0], Inputs[i][j][6] - \
    #                  gyoCalStc[2][0]
    #     data = smooth.getSmoothResult([ax, ay, az, gx, gy, gz])
    #     G = np.array([[data[3]], [data[4]], [data[5]]])
    #     Q = updateQ(np.deg2rad(Truths[i][0][1:4].reshape(3, 1)), np.deg2rad(Truths[i][0][4:7].reshape(3, 1)), G, Cbn, Q)
    #     ma.setQ(Q)
    #     # ma.MahonyAHRSupdateIMU(gx, gy, gz, ax, ay, az)
    #     Q = ma.getQ()
    #     Cbn = getRMatrix(Q)
    #     Eular[0] = degrees(atan2(Cbn[2, 1], Cbn[2, 2]))
    #     Eular[1] = degrees(atan2(-Cbn[2, 0], sqrt(Cbn[2, 1] * Cbn[2, 1] + Cbn[2, 2] * Cbn[2, 2])))
    #     Eular[2] = -degrees(atan2(Cbn[1, 0], Cbn[0, 0]))
    #     if Eular[2] < 0:
    #         Eular[2] += 360
    #
    # print("真值欧拉角：" + str(Truths[i][1][7]) + "," + str(Truths[i][1][8]) + "," + str(Truths[i][1][9]))
    # print("计算欧拉角：" + str(Eular[0]) + "," + str(Eular[1]) + "," + str(Eular[2]))
    # roll_error = abs(Truths[i][1][7] - Eular[0])
    # if roll_error > 180:
    #     roll_error = 360 - roll_error
    # pitch_error = abs(Truths[i][1][8] - Eular[1])
    # if pitch_error > 180:
    #     pitch_error = 360 - pitch_error
    # yaw_error = abs(Truths[i][1][9] - Eular[2])
    # if yaw_error > 180:
    #     yaw_error = 360 - yaw_error
    # Eular_error.append([roll_error, pitch_error, yaw_error])
    # print("------------------------------------------------------------------------------------------")

    for j in range(Inputs.shape[1]):
        ax, ay, az = Inputs[i][j][1] - accCalStc[0][0], Inputs[i][j][2] - accCalStc[1][0], Inputs[i][j][3] - \
                     accCalStc[2][0]
        gx, gy, gz = Inputs[i][j][4] - gyoCalStc[0][0], Inputs[i][j][5] - gyoCalStc[1][0], Inputs[i][j][6] - \
                     gyoCalStc[2][0]
        # 更新四元数及姿态矩阵
        # ma.MahonyAHRSupdateIMU(gx, gy, gz, ax, ay, az)
        data = smooth.getSmoothResult([ax, ay, az, gx, gy, gz])
        G = np.array([[data[3]], [data[4]], [data[5]]])
        Q = updateQ(P_INS, V_INS, G, Cbn, Q)
        ma.setQ(Q)
        Cbn = getRMatrix(Q)
        # 姿态更新完毕
        # 计算加速度信息
        a_b = np.array([[ax], [ay], [az]])
        w_n_ie = np.array([[0], [wie * cos(P_INS[1][0])], [wie * sin(P_INS[1][0])]])
        w_n_en = np.array([[-V_INS[1][0] / Rm], [V_INS[0][0] / Rn], [V_INS[0][0] / Rn * tan(P_INS[1][0])]])
        W = np.mat([[0, 2 * w_n_ie[2][0] + w_n_en[2][0], -(2 * w_n_ie[1][0] + w_n_en[1][0])],
                    [-(2 * w_n_ie[2][0] + w_n_en[2][0]), 0, 2 * w_n_ie[0][0] + w_n_en[0][0]],
                    [2 * w_n_ie[1][0] + w_n_en[1][0], -(2 * w_n_ie[0][0] + w_n_en[0][0]), 0]])
        a_n = Cbn * a_b - W * V_INS + g
        # 加速度计算完毕
        # 更新速度
        V_INS += a_n / sampleFreq
        # 速度更新完毕
        # 更新经纬度
        P_INS += np.array([[V_INS[0][0] / (cos(P_INS[1][0]) * Rn) / sampleFreq],
                           [V_INS[1][0] / Rm / sampleFreq],
                           [V_INS[2][0] / sampleFreq]])
        # 经纬度更新完毕
    # 进行KF计算
    if States[i][1][4] < 3.0:
        V_GPS = np.array([[Truths[i][1][4]], [Truths[i][1][5]], [Truths[i][1][6]]])
        P_GPS = np.array([[radians(States[i][1][1])], [radians(States[i][1][2])], [States[i][1][3]]])
        Dpv = np.mat(np.concatenate((P_INS - P_GPS, V_INS - V_GPS), axis=0))
        XX = KF.kalman_GPS_INS_pv(Dpv, V_GPS, P_GPS, Cbn, a_n, 1, Rm, Rn)

        V_INS -= np.array([[XX[3, 0]], [XX[4, 0]], [XX[5, 0]]])
        P_INS -= np.array([[XX[6, 0]], [XX[7, 0]], [XX[8, 0]]])
        gyoCalStc = np.array([[XX[9, 0]], [XX[10, 0]], [XX[11, 0]]]) - \
                    np.array([[XX[12, 0]], [XX[13, 0]], [XX[14, 0]]]) / 300
        accCalStc = np.array([[XX[15, 0]], [XX[16, 0]], [XX[17, 0]]]) / 1000
        Cnn = np.mat([[1, -XX[2, 0], XX[1, 0]],
                      [XX[2, 0], 1, -XX[0, 0]],
                      [-XX[1, 0], XX[0, 0], 1]])
        Cbn = Cbn * Cnn
        Eular[0] = atan2(Cbn[2, 1], Cbn[2, 2])
        Eular[1] = atan2(-Cbn[2, 0], sqrt(Cbn[2, 1] * Cbn[2, 1] + Cbn[2, 2] * Cbn[2, 2]))
        Eular[2] = atan2(Cbn[1, 0], Cbn[0, 0])
        Q = [cos(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
             + sin(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2),
             cos(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2)
             - sin(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2),
             cos(Eular[2] / 2) * sin(Eular[1] / 2) * cos(Eular[0] / 2)
             + sin(Eular[2] / 2) * cos(Eular[1] / 2) * sin(Eular[0] / 2),
             sin(Eular[2] / 2) * cos(Eular[1] / 2) * cos(Eular[0] / 2)
             - cos(Eular[2] / 2) * sin(Eular[1] / 2) * sin(Eular[0] / 2)]
        ma.setQ(Q)
    print("第" + str(i) + "次计算结果：")
    print("真值经纬度：" + str(States[i][1][1]) + "," + str(States[i][1][2]))
    print("计算经纬度：" + str(degrees(P_INS[0][0])) + "," + str(degrees(P_INS[1][0])))
    print("距离误差：" + str(distance_by_LoLa(States[i][1][1], States[i][1][2], degrees(P_INS[0][0]), degrees(P_INS[1][0]))))
    print("东向距离误差：" + str(distance_by_LoLa(States[i][1][1], States[i][1][2], degrees(P_INS[0][0]), States[i][1][2])))
    print("北向距离误差：" + str(distance_by_LoLa(States[i][1][1], States[i][1][2], States[i][1][1], degrees(P_INS[1][0]))))
    print("------------------------------------------------------------------------------------------")
    # KF计算完毕
# for i in Eular_error:
#     csv_writer.writerow(i)
