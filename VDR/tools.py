import math
import numpy as np


# 根据经纬度计算距离
def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = math.radians(La1)
    radLa2 = math.radians(La2)
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * math.asin(math.sqrt(
        math.pow(math.sin(deltaLa / 2), 2) +
        math.cos(radLa1) * math.cos(radLa2) * math.pow(math.sin(deltaLo / 2), 2))) * Er


# NMEA0183中的经纬度数据转换成实际的经纬度数据
def NMEA2LL(nmea_data):
    if nmea_data == '':
        return nmea_data
    data = float(nmea_data) / 100
    pointfront = int(data)
    pointback = (data - pointfront) * 100
    pointback *= 0.0166667
    data = pointfront + pointback
    return data


# 根据上一时刻的速度和加速度，以及时间间隔，计算该事件间隔内的位移
def getposition(v, a, LENGTH_EPOCH):
    return v / LENGTH_EPOCH + 0.5 * a / LENGTH_EPOCH / LENGTH_EPOCH


# 根据四元数和三轴加速度，获取旋转后的三轴加速度
def getRMatrix(q0, q1, q2, q3, ax, ay, az):
    Matrix_R = np.array(
        [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
         [2 * (q0 * q3 + q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
         [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
    )
    Matrix_A = np.dot(Matrix_R, [ax, ay, az]).tolist()
    return Matrix_A


# 根据两点的经纬度，计算东向和北向位移
def getEN_Pos(lo1, la1, lo2, la2):
    E_pos = distance_by_LoLa(lo1, la1, lo2, la1)
    N_pos = distance_by_LoLa(lo1, la1, lo1, la2)
    if lo2 - lo1 < 0:
        E_pos *= -1
    if la2 - la1 < 0:
        N_pos *= -1
    return E_pos, N_pos


# 根据重力加速度和地磁计信息，计算旋转矩阵
def getRotationMatrix(gravity, magnetic):
    Ax, Ay, Az = gravity[0], gravity[1], gravity[2]
    normsqA = Ax * Ax + Ay * Ay + Az * Az
    g = 9.81
    freeFallGravitySquared = 0.01 * g * g
    if normsqA < freeFallGravitySquared:
        return False

    Ex, Ey, Ez = magnetic[0], magnetic[1], magnetic[2]
    Hx, Hy, Hz = Ey * Az - Ez * Ay, Ez * Ax - Ex * Az, Ex * Ay - Ey * Ax
    normH = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)

    if normH < 0.1:
        return False

    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz *= invH
    invA = 1.0 / math.sqrt(normsqA)
    Ax *= invA
    Ay *= invA
    Az *= invA

    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz
    Mz = Ax * Hy - Ay * Hx

    R = np.array([[Hx, Hy, Hz],
                  [Mx, My, Mz],
                  [Ax, Ay, Az]])
    return R