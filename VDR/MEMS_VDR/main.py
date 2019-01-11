import csv
from VDR.MEMS_VDR.tool import *
import VDR.MEMS_VDR.Parameter as glv
import matplotlib.pyplot as plt

deg2rad = pi / 180
rad2deg = 180 / pi

earthRe, earthr, earthf, earthe, earthwie = 6378137, 6356752.3142, 1 / 298.257, 0.0818, 7.292e-5
Rm, Rn = 0.0, 0.0

DIR = r'D:\Project\VisualStudio\PSINS\psins_vs2017'
InputFile = DIR + r'\data1 - 副本.csv'
OutputFile = DIR + r'\py_output.csv'
out = open(OutputFile, 'w', newline='')
csv_reader = csv.reader(open(InputFile, 'r'))
csv_writer = csv.writer(out)

index = 0
datas = []
for row in csv_reader:
    if index == 0:
        index += 1
        continue
    else:
        datas.append(list(map(float, row)))

ts = 0.01
isStart = False

# 姿态四元数和旋转矩阵
Qbn = np.array([[0.0]] * 4)
Cbn = np.mat([[0.0] * 3] * 3)

# 初始位置和速度
pos = np.array([[0.0]] * 3)
vn = np.array([[0.0]] * 3)

# 陀螺仪和加计零偏
eb = np.array([[0.0]] * 3)
db = np.array([[0.0]] * 3)

# n系下加速度信息
an = np.array([[0.0]] * 3)
# 地球参数
eth = CEarth()

# KF相关参数
Ft = np.mat([[0.0] * 15] * 15)
Xk = np.mat([[0.0]] * 15)
Pk = np.mat([[0.0] * 15] * 15)
Qt = np.mat([[0.0] * 15] * 15)
Rt = np.mat([[0.0] * 6] * 6)
Zk = np.mat([[0.0]] * 6)
Hk = np.mat([[0.0] * 15] * 6)
Hk[0, 3] = 1.0
Hk[1, 4] = 1.0
Hk[2, 5] = 1.0
Hk[3, 6] = 1.0
Hk[4, 7] = 1.0
Hk[5, 8] = 1.0


def SetFt(vn_, pos_, fn_, Cbn_, wie_):
    global Ft
    Ve, Vn, Vu = vn_[0, 0], vn_[1, 0], vn_[2, 0]
    L, E, h = pos_[0, 0], pos_[1, 0], pos_[2, 0]
    sl, cl, tl = sin(L), cos(L), tan(L)
    secl = 1 / cl
    Fe, Fn, Fu = fn_[0, 0], fn_[1, 0], fn_[2, 0]
    Rnh, Rmh = Rn + h, Rm + h
    Ft[0, 0] = 0.0
    Ft[0, 1] = wie_ * sl + Ve * tl / Rnh
    Ft[0, 2] = -(wie_ * cl + Ve / Rnh)
    Ft[0, 3] = 0.0
    Ft[0, 4] = -1.0 / Rmh
    Ft[0, 5] = 0.0
    Ft[0, 9] = Cbn_[0, 0]
    Ft[0, 10] = Cbn_[0, 1]
    Ft[0, 11] = Cbn_[0, 2]

    Ft[1, 0] = -(wie_ * sl + Ve * tl / Rnh)
    Ft[1, 1] = 0.0
    Ft[1, 2] = -Vn / Rmh
    Ft[1, 3] = 1.0 / Rnh
    Ft[1, 4] = 0.0
    Ft[1, 5] = 0.0
    Ft[1, 6] = -wie_ * sl
    Ft[1, 9] = Cbn_[1, 0]
    Ft[1, 10] = Cbn_[1, 1]
    Ft[1, 11] = Cbn_[1, 2]

    Ft[2, 0] = wie_ * cl + Ve / Rnh
    Ft[2, 1] = Vn / Rmh
    Ft[2, 2] = 0.0
    Ft[2, 3] = tl / Rnh
    Ft[2, 4] = 0.0
    Ft[2, 5] = 0.0
    Ft[2, 6] = wie_ * cl + Ve / Rnh * (secl ** 2)
    Ft[2, 9] = Cbn_[2, 0]
    Ft[2, 10] = Cbn_[2, 1]
    Ft[2, 11] = Cbn_[2, 2]

    Ft[3, 0] = 0.0
    Ft[3, 1] = -Fu
    Ft[3, 2] = Fn
    Ft[3, 3] = (Vn * tl - Vu) / Rnh
    Ft[3, 4] = 2 * wie_ * sl + Ve * tl / Rnh
    Ft[3, 5] = -(2 * wie_ * cl + Ve / Rnh)
    Ft[3, 6] = 2 * wie_ * (cl * Vn + sl * Vu) + Ve * Vn * (secl ** 2) / Rnh
    Ft[3, 12] = Cbn_[0, 0]
    Ft[3, 13] = Cbn_[0, 1]
    Ft[3, 14] = Cbn_[0, 2]

    Ft[4, 0] = Fu
    Ft[4, 1] = 0.0
    Ft[4, 2] = -Fe
    Ft[4, 3] = -2 * (wie_ * sl + Ve * tl / Rnh)
    Ft[4, 4] = -Vu / Rmh
    Ft[4, 5] = -Vn / Rmh
    Ft[4, 6] = -(2 * wie_ * cl * Ve + (Ve * secl) ** 2 / Rnh)
    Ft[4, 12] = Cbn_[1, 0]
    Ft[4, 13] = Cbn_[1, 1]
    Ft[4, 14] = Cbn_[1, 2]

    Ft[5, 0] = -Fn
    Ft[5, 1] = Fe
    Ft[5, 2] = 0.0
    Ft[5, 3] = 2 * (wie_ * cl + Ve / Rnh)
    Ft[5, 4] = 2 * Vn / Rmh
    Ft[5, 5] = 0.0
    Ft[5, 6] = -2 * wie_ * sl * Ve
    Ft[5, 12] = Cbn_[2, 0]
    Ft[5, 13] = Cbn_[2, 1]
    Ft[5, 14] = Cbn_[2, 2]

    Ft[6, 4] = 1.0 / Rmh

    Ft[7, 3] = secl / Rnh
    Ft[7, 6] = Ve * secl * tl / Rnh

    Ft[8, 5] = 1.0

    Ft[9, 9] = -1 / 300
    Ft[10, 10] = -1 / 300
    Ft[11, 11] = -1 / 300
    Ft[12, 12] = -1 / 1000
    Ft[13, 13] = -1 / 1000
    Ft[14, 14] = -1 / 1000


def Init_KF():
    global Pk, Qt, Rt
    Pk = np.diagflat(
        np.square(
            [1.0 * glv.deg, 1.0 * glv.deg, 30.0 * glv.deg, 1.0, 1.0, 1.0, 100.0 / glv.Re, 100.0 / glv.Re, 100.0,
             100.0 * glv.dph, 100.0 * glv.dph, 100.0 * glv.dph, 10.0 * glv.mg, 10.0 * glv.mg, 10.0 * glv.mg]
        )
    )
    Qt = np.diagflat(
        np.square(
            [1.0 * glv.dpsh, 1.0 * glv.dpsh, 1.0 * glv.dpsh, 100.0 * glv.ugpsHz, 100.0 * glv.ugpsHz, 100.0 * glv.ugpsHz,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    )
    Rt = np.diagflat(
        np.square(
            [0.5, 0.5, 0.5, 10.0 / glv.Re, 10.0 / glv.Re, 10.0]
        )
    )


Error = []
Time = []
for i in range(len(datas)):
    print('第', i, '次计算')
    gyo = np.array([radians(datas[i][1]), radians(datas[i][2]), radians(datas[i][3])]).reshape(-1, 1)  # rad / s
    acc = np.array([datas[i][4], datas[i][5], datas[i][6]]).reshape(-1, 1)  # m/s^2
    gyo = gyo - eb
    acc = acc - db
    mag = np.array([datas[i][25], datas[i][26], datas[i][27]]).reshape(-1, 1)  # mG
    eular = np.array([datas[i][7], datas[i][8], datas[i][9]]).reshape(-1, 1)  # deg
    GPSVn = np.array([datas[i][16], datas[i][17], datas[i][18]]).reshape(-1, 1)  # m/s
    GPSPos = np.array([radians(datas[i][19]), radians(datas[i][20]), datas[i][21]]).reshape(-1, 1)  # rad
    GPSYaw = datas[i][22]
    if GPSYaw < 0:
        GPSYaw = 360 + GPSYaw
    GPSYaw = radians(GPSYaw)
    GPSPDOP = datas[i][23]

    wm = gyo * ts
    vm = acc * ts

    if not isStart:
        Qbn = A2Q(eular * deg2rad)
        Cbn = Q2M(Qbn)

    # 表示已经接受过GNSS信号了，可以开始工作了
    if isStart:
        # 更新速度和位置
        vn01 = vn + an * ts / 2
        pos01 = np.array([[0.0]] * 3)
        pos01[0, 0] = pos[0, 0] + (vn01[1, 0] / Rm) * (ts / 2)
        pos01[1, 0] = pos[1, 0] + (vn01[0, 0] / (cos(pos01[0, 0]) * Rn)) * (ts / 2)
        pos01[2, 0] = pos[2, 0] + vn01[2, 0] * (ts / 2)
        Rm = earthRe * (1 - 2 * earthf + 3 * earthf * sin(pos01[0, 0]) * sin(pos01[0, 0]))
        Rn = earthRe * (1 + earthf * sin(pos01[0, 0]) * sin(pos01[0, 0]))
        wib = gyo
        fb = acc
        wnie = np.array([[0.0]] * 3)
        wnen = np.array([[0.0]] * 3)
        wnie[0, 0], wnie[1, 0], wnie[2, 0] = 0.0, glv.wie * cos(pos[0, 0]), glv.wie * sin(pos[0, 0])
        wnen[0, 0], wnen[1, 0] = -vn01[1, 0] * 1 / Rm, vn01[0, 0] * 1 / Rn
        wnen[2, 0] = wnen[1, 0] * tan(pos[0, 0])
        g = -(glv.g0 * (1 + 5.27094e-3 * sin(pos[0, 0]) ** 2 + 2.32718e-5 * sin(pos[0, 0]) ** 4) - 3.086e-6 * pos[2, 0])
        g = np.array([[0.0], [0.0], [g]])
        # 根据姿态角计算姿态矩阵
        Qbn = A2Q(eular * deg2rad)
        Cbn = Q2M(Qbn)

        fn = Cbn * fb
        wnin = np.mat([[0, 2 * wnie[2, 0] + wnen[2, 0], -(2 * wnie[1, 0] + wnen[1, 0])],
                       [-(2 * wnie[2, 0] + wnen[2, 0]), 0, 2 * wnie[0, 0] + wnen[0, 0]],
                       [2 * wnie[1, 0] + wnen[1, 0], -(2 * wnie[0, 0] + wnen[0, 0]), 0]
                       ])
        an = fn + g + wnin * vn
        vn1 = vn + an * ts
        pos[0, 0] = pos[0, 0] + (vn1[1, 0] / Rm) * ts
        pos[1, 0] = pos[1, 0] + (vn1[0, 0] / (cos(pos[0, 0]) * Rn)) * ts
        pos[2, 0] = pos[2, 0] + vn1[2, 0] * ts
        vn = vn1

        # 状态方程更新
        SetFt(vn, pos, fn, Cbn, glv.wie)
        Fk = np.eye(15) + Ft * ts
        Xk = Fk * Xk
        Pk = Fk * Pk * Fk.T
        Pk += Qt * ts

    if GPSPos[0, 0] != 0 and GPSPos[1, 0] != 0:
        if not isStart:
            # 第一次接受到GNSS信号，确定初始位置和初始速度等
            isStart = True
            pos = GPSPos
            vn = GPSVn
            Rm = earthRe * (1 - 2 * earthf + 3 * earthf * sin(pos[0, 0]) * sin(pos[0, 0]))
            Rn = earthRe * (1 + earthf * sin(pos[0, 0]) * sin(pos[0, 0]))
            Init_KF()
        else:
            # KF更新
            delta_V = vn - GPSVn
            delta_P = pos - GPSPos
            Zk = np.vstack((delta_V, delta_P))
            Pkh = Hk * Pk * Hk.T
            Kk = Pk * Hk.T * (Pkh + Rt).I
            Xk = Xk + Kk * (Zk - Hk * Xk)
            Pk = Pk - Kk * Hk * Pk
            Pk = (Pk + Pk.T) * 0.5

    # 反馈校正
    eb += np.array([[Xk[9, 0]], [Xk[10, 0]], [Xk[11, 0]]])
    db += np.array([[Xk[12, 0]], [Xk[13, 0]], [Xk[14, 0]]])
    pos -= np.array([[Xk[6, 0]], [Xk[7, 0]], [Xk[8, 0]]])
    vn -= np.array([[Xk[3, 0]], [Xk[4, 0]], [Xk[5, 0]]])
    Qbn -= rv2Q(np.array([[Xk[0, 0]], [Xk[1, 0]], [Xk[2, 0]]]))
    Xk = np.mat([[0.0]] * 15)

    if GPSPos[0, 0] != 0 and GPSPos[1, 0] != 0:
        data2write = [datas[i][0], pos[0, 0] * rad2deg, pos[1, 0] * rad2deg, pos[2, 0], GPSPos[0, 0] * rad2deg,
                      GPSPos[1, 0] * rad2deg, GPSPos[2, 0]]
        csv_writer.writerow(data2write)
        Error.append(
            distance_by_LoLa(pos[1, 0] * rad2deg, pos[0, 0] * rad2deg, GPSPos[1, 0] * rad2deg, GPSPos[0, 0] * rad2deg))
        Time.append(datas[i][0])

plt.plot(Time, Error, '-', linewidth=2)
plt.show()
out.close()
