from math import *
import numpy as np
import VDR.MEMS_VDR.Parameter as glv

conefactors = [[2.0 / 3.0, 0.0, 0.0, 0.0],
               [9.0 / 20.0, 27.0 / 20.0, 0.0, 0.0],
               [54.0 / 105.0, 92.0 / 105.0, 214.0 / 105.0, 0.0],
               [250.0 / 504.0, 525.0 / 504.0, 650.0 / 504.0, 1375.0 / 504.0],
               [0.0, 0.0, 0.0, 0.0]]


def M2Q(Cnb):
    q0, q1, q2, q3, qq4 = 0.0, 0.0, 0.0, 0.0, 0.0
    if Cnb[0, 0] >= Cnb[1, 1] + Cnb[2, 2]:
        q1 = 0.5 * sqrt(1 + Cnb[0, 0] - Cnb[1, 1] - Cnb[2, 2])
        qq4 = 4 * q1
        q0 = (Cnb[2, 1] - Cnb[1, 2]) / qq4
        q2 = (Cnb[0, 1] + Cnb[1, 0]) / qq4
        q3 = (Cnb[0, 2] + Cnb[2, 0]) / qq4
    elif Cnb[1, 1] >= Cnb[0, 0] + Cnb[2, 2]:
        q2 = 0.5 * sqrt(1 - Cnb[0, 0] + Cnb[1, 1] - Cnb[2, 2])
        qq4 = 4 * q2
        q0 = (Cnb[0, 2] - Cnb[2, 0]) / qq4
        q1 = (Cnb[0, 1] - Cnb[1, 0]) / qq4
        q3 = (Cnb[1, 2] + Cnb[2, 1]) / qq4
    elif Cnb[2, 2] >= Cnb[0, 0] + Cnb[1, 1]:
        q3 = 0.5 * sqrt(1 - Cnb[0, 0] - Cnb[1, 1] + Cnb[2, 2])
        qq4 = 4 * q3
        q0 = (Cnb[1, 0] - Cnb[0, 1]) / qq4
        q1 = (Cnb[0, 2] + Cnb[2, 0]) / qq4
        q2 = (Cnb[1, 2] + Cnb[2, 1]) / qq4
    else:
        q0 = 0.5 * sqrt(1 + Cnb[0, 0] + Cnb[1, 1] + Cnb[2, 2])
        qq4 = 4 * q2
        q1 = (Cnb[2, 1] - Cnb[1, 2]) / qq4
        q2 = (Cnb[0, 2] - Cnb[2, 0]) / qq4
        q3 = (Cnb[1, 0] - Cnb[0, 1]) / qq4
    nq = sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    q0 /= nq
    q1 /= nq
    q2 /= nq
    q3 /= nq
    return np.array([q0, q1, q2, q3]).reshape(4, 1)


def A2Q(Eular):
    pitch, roll, yaw = Eular[0, 0] / 2, Eular[1, 0] / 2, Eular[2, 0] / 2
    sp, sr, sy = sin(pitch), sin(roll), sin(yaw)
    cp, cr, cy = cos(pitch), cos(roll), cos(yaw)
    q0 = cp * cr * cy - sp * sr * sy
    q1 = sp * cr * cy - cp * sr * sy
    q2 = cp * sr * cy + sp * cr * sy
    q3 = cp * cr * sy + sp * sr * cy
    return np.array([q0, q1, q2, q3]).reshape(4, 1)


def Q2M(Q):
    q11, q12, q13, q14 = Q[0, 0] * Q[0, 0], Q[0, 0] * Q[1, 0], Q[0, 0] * Q[2, 0], Q[0, 0] * Q[3, 0]
    q22, q23, q24 = Q[1, 0] * Q[1, 0], Q[1, 0] * Q[2, 0], Q[1, 0] * Q[3, 0]
    q33, q34 = Q[2, 0] * Q[2, 0], Q[2, 0] * Q[3, 0]
    q44 = Q[3, 0] * Q[3, 0]

    return np.mat([[q11 + q22 - q33 - q44, 2 * (q23 - q14), 2 * (q24 + q13)],
                   [2 * (q23 + q14), q11 - q22 + q33 - q44, 2 * (q34 - q12)],
                   [2 * (q24 - q13), 2 * (q34 + q12), q11 - q22 - q33 + q44]])


def M2A(Cnb):
    i = asinEx(Cnb[2, 1])
    j = atan2Ex(-Cnb[2, 0], Cnb[2, 2])
    k = atan2Ex(-Cnb[0, 1], Cnb[1, 1])
    return np.array([[i], [j], [k]])


def p_range(val, min, max):
    if val > max:
        val = max
    elif val < min:
        val = min
    return val


def atan2Ex(y, x):
    if sign(y) == 0 and sign(x) == 0:
        return 0.0
    else:
        return atan2(y, x)


def sign(x):
    if x < 2.220446049e-16 * -1:
        return -1
    elif x > 2.220446049e-16:
        return 1
    else:
        return 0


def asinEx(x):
    return asin(p_range(x, -1, 1))


def askew(v):
    return np.mat([[0.0, -v[2, 0], v[1, 0]],
                   [v[2, 0], 0.0, -v[0, 0]],
                   [-v[1, 0], v[0, 0], 0.0]])


def cross(v1, v2):
    return np.array([[v1[1, 0] * v2[2, 0] - v1[2, 0] * v2[1, 0]],
                     [v1[2, 0] * v2[0, 0] - v1[0, 0] * v2[2, 0]],
                     [v1[0, 0] * v2[1, 0] - v1[1, 0] * v2[0, 0]]
                     ])


def Q_cross(Q1, Q2):
    q0 = Q1[0, 0] * Q2[0, 0] - Q1[1, 0] * Q2[1, 0] - Q1[2, 0] * Q2[2, 0] - Q1[3, 0] * Q2[3, 0]
    q1 = Q1[0, 0] * Q2[1, 0] + Q1[1, 0] * Q2[0, 0] + Q1[2, 0] * Q2[3, 0] - Q1[3, 0] * Q2[2, 0]
    q2 = Q1[0, 0] * Q2[2, 0] + Q1[2, 0] * Q2[0, 0] + Q1[3, 0] * Q2[1, 0] - Q1[1, 0] * Q2[3, 0]
    q3 = Q1[0, 0] * Q2[3, 0] + Q1[3, 0] * Q2[0, 0] + Q1[1, 0] * Q2[2, 0] - Q1[2, 0] * Q2[1, 0]
    return np.array([q0, q1, q2, q3]).reshape(4, 1)


def QV_cross(Q, v):
    q0 = -Q[1, 0] * v[0, 0] - Q[2, 0] * v[1, 0] - Q[3, 0] * v[2, 0]
    q1 = Q[0, 0] * v[0, 0] + Q[2, 0] * v[2, 0] - Q[3, 0] * v[1, 0]
    q2 = Q[0, 0] * v[1, 0] + Q[3, 0] * v[0, 0] - Q[1, 0] * v[2, 0]
    q3 = Q[0, 0] * v[2, 0] + Q[1, 0] * v[1, 0] - Q[2, 0] * v[0, 0]

    i = -q0 * Q[1, 0] + q1 * Q[0, 0] - q2 * Q[3, 0] + q3 * Q[2, 0]
    j = -q0 * Q[2, 0] + q2 * Q[0, 0] - q3 * Q[1, 0] + q1 * Q[3, 0]
    k = -q0 * Q[3, 0] + q3 * Q[0, 0] - q1 * Q[2, 0] + q2 * Q[1, 0]

    return np.array([i, j, k]).reshape(3, 1)


def rv2Q(rv):
    F1 = 2.0 * 1.0
    F2 = F1 * 2 * 2
    F3 = F2 * 2 * 3
    F4 = F3 * 2 * 4
    F5 = F4 * 2 * 5
    n2 = rv[0, 0] * rv[0, 0] + rv[1, 0] * rv[1, 0] + rv[2, 0] * rv[2, 0]
    if n2 < pi / 180 * pi / 180:
        n4 = n2 * n2
        c = 1.0 - n2 * (1.0 / F2) + n4 * (1.0 / F4)
        f = 0.5 - n2 * (1.0 / F3) + n4 * (1.0 / F5)
    else:
        n_2 = sqrt(n2) / 2.0
        c = cos(n_2)
        f = sin(n_2) / n_2 * 0.5
    return np.array([c, f * rv[0, 0], f * rv[1, 0], f * rv[2, 0]]).reshape(4, 1)


def symmetry(m):
    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            m[i, j] = m[j, i] = (m[i, j] + m[j, i]) * 0.5
    return m


def pos2Cen(pos):
    si = sin(pos[0, 0])
    ci = cos(pos[0, 0])
    sj = sin(pos[1, 0])
    cj = cos(pos[1, 0])
    return np.mat([[-sj, -si * cj, ci * cj],
                   [cj, -si * sj, ci * sj],
                   [0, ci, si]])


def AlignCoarse(wmm, vmm, latitude):
    cl = cos(latitude)
    tl = tan(latitude)
    wbib = wmm / norm(wmm)
    fb = vmm / norm(vmm)
    T31 = fb[0, 0]
    T32 = fb[1, 0]
    T33 = fb[2, 0]
    T21 = wbib[0, 0] / cl - T31 * tl
    T22 = wbib[1, 0] / cl - T32 * tl
    T23 = wbib[2, 0] / cl - T33 * tl
    nn = sqrt(T21 * T21 + T22 * T22 + T23 * T23)
    T21 /= nn
    T22 /= nn
    T23 /= nn
    T11 = T22 * T33 - T23 * T32
    T12 = T23 * T31 - T21 * T33
    T13 = T21 * T32 - T22 * T31
    sqrt(T11 * T11 + T12 * T12 + T13 * T13)
    T11 /= nn
    T12 /= nn
    T13 /= nn
    return np.array([[asinEx(T32)], [atan2Ex(-T31, T33)], [atan2Ex(-T12, T22)]])


def norm(v):
    return sqrt(v[0, 0] * v[0, 0] + v[1, 0] * v[1, 0] + v[2, 0] * v[2, 0])


def Vec2Mat(att):
    si = sin(att[0, 0])
    ci = cos(att[0, 0])
    sj = sin(att[1, 0])
    cj = cos(att[1, 0])
    sk = sin(att[2, 0])
    ck = cos(att[2, 0])

    e00 = cj * ck - si * sj * sk
    e01 = -ci * sk
    e02 = sj * ck + si * cj * sk
    e10 = cj * sk + si * sj * ck
    e11 = ci * ck
    e12 = sj * sk - si * cj * ck
    e20 = -ci * sj
    e21 = si
    e22 = ci * cj

    return np.mat([[e00, e01, e02],
                   [e10, e11, e12],
                   [e20, e21, e22]])


def Mat2Qua(Cnb):
    tmp = 1.0 + Cnb[0, 0] - Cnb[1, 1] - Cnb[2, 2]
    q1 = sqrt(fabs(tmp)) / 2.0
    tmp = 1.0 - Cnb[0, 0] + Cnb[1, 1] - Cnb[2, 2]
    q2 = sqrt(fabs(tmp)) / 2.0
    tmp = 1.0 - Cnb[0, 0] - Cnb[1, 1] + Cnb[2, 2]
    q3 = sqrt(fabs(tmp)) / 2.0
    tmp = 1.0 - q1 * q1 - q2 * q2 - q3 * q3
    q0 = sqrt(fabs(tmp))
    if Cnb[2, 1] - Cnb[1, 2] < 0:
        q1 = -q1
    if Cnb[0, 2] - Cnb[2, 0] < 0:
        q2 = -q2
    if Cnb[1, 0] - Cnb[0, 1] < 0:
        q3 = -q3

    nq = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    q0 /= nq
    q1 /= nq
    q2 /= nq
    q3 /= nq
    return np.array([[q0], [q1], [q2], [q3]])


def dv2att(va1, va2, vb1, vb2):
    a = cross(va1, va2)
    b = cross(vb1, vb2)
    aa = cross(a, va1)
    bb = cross(b, vb1)
    va1 = va1 / norm(va1)
    a = a / norm(a)
    aa = aa / norm(aa)
    vb1 = vb1 / norm(vb1)
    b = b / norm(b)
    bb = bb / norm(bb)

    Ma = np.mat([[va1[0, 0], va1[1, 0], va1[2, 0]],
                 [a[0, 0], a[1, 0], a[2, 0]],
                 [aa[0, 0], aa[1, 0], aa[2, 0]]])

    Mb = np.mat([[vb1[0, 0], vb1[1, 0], vb1[2, 0]],
                 [b[0, 0], b[1, 0], b[2, 0]],
                 [bb[0, 0], bb[1, 0], bb[2, 0]]])

    return Ma.T * Mb


# 根据经纬度计算距离
def distance_by_LoLa(Lo1, La1, Lo2, La2):
    Er = 6378137.0
    radLa1 = radians(La1)
    radLa2 = radians(La2)
    deltaLa = radLa1 - radLa2
    deltaLo = (Lo1 - Lo2) * np.pi / 180.0
    return 2 * asin(sqrt(pow(sin(deltaLa / 2), 2) + cos(radLa1) * cos(radLa2) * pow(sin(deltaLo / 2), 2))) * Er


class CEarth(object):
    a = b = f = e = e2 = ep = ep2 = wie = sl = sl2 = sl4 = cl = tl = RMh = RNh = clRNh = f_RMh = f_RNh = f_clRNh = 0.0
    pos = np.array([[0.0]] * 3)
    vn = np.array([[0.0]] * 3)
    wnie = np.array([[0.0]] * 3)
    wnen = np.array([[0.0]] * 3)
    wnin = np.array([[0.0]] * 3)
    gn = np.array([[0.0]] * 3)
    gcc = np.array([[0.0]] * 3)

    def __init__(self, a0=glv.Re, f0=glv.f, g0=glv.g0):
        self.a = a0
        self.f = f0
        self.wie = glv.wie
        self.b = (1 - self.f) * self.a
        self.e = sqrt(self.a * self.a - self.b * self.b) / self.a
        self.e2 = self.e * self.e
        self.gn = np.array([[0], [0], [-glv.g0]])

    def Update(self, pos, vn=np.array([[0.0]] * 3)):
        self.pos = pos
        self.vn = vn
        self.sl, self.cl = sin(pos[0, 0]), cos(pos[0, 0])
        self.tl = self.sl / self.cl
        sq = 1 - self.e2 * self.sl * self.sl
        sq2 = sqrt(sq)
        self.RMh = self.a * (1 - self.e2) / sq / sq2 + pos[2, 0]
        self.f_RMh = 1.0 / self.RMh
        self.RNh = self.a / sq2 + pos[2, 0]
        self.clRNh = self.cl * self.RNh
        self.f_RNh = 1.0 / self.RNh
        self.f_clRNh = 1.0 / self.clRNh
        self.wnie[0, 0], self.wnie[1, 0], self.wnie[2, 0] = 0.0, self.wie * self.cl, self.wie * self.sl
        self.wnen[0, 0], self.wnen[1, 0] = -vn[1, 0] * self.f_RMh, vn[0, 0] * self.f_RNh,
        self.wnen[2, 0] = self.wnen[1, 0] * self.tl
        self.wnin = self.wnie + self.wnen
        self.sl2 = self.sl * self.sl
        self.sl4 = self.sl2 * self.sl2
        self.gn[2, 0] = -(glv.g0 * (1 + 5.27094e-3 * self.sl2 + 2.32718e-5 * self.sl4) - 3.086e-6 * pos[2, 0])
        self.gcc = self.gn - cross((self.wnie + self.wnin), vn)

    def vn2pos(self, vn, ts=1.0):
        return np.array([[vn[1, 0] * self.f_RMh],
                         [vn[0, 0] * self.f_clRNh],
                         [vn[2, 0]]]) * ts
