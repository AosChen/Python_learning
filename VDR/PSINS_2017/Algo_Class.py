from VDR.PSINS_2017.tool import *
import VDR.PSINS_2017.Parameter as glv

conefactors = [[2.0 / 3.0, 0.0, 0.0, 0.0],
               [9.0 / 20.0, 27.0 / 20.0, 0.0, 0.0],
               [54.0 / 105.0, 92.0 / 105.0, 214.0 / 105.0, 0.0],
               [250.0 / 504.0, 525.0 / 504.0, 650.0 / 504.0, 1375.0 / 504.0],
               [0.0, 0.0, 0.0, 0.0]]

EPS = float(2.220446049e-16)
INF = float(3.402823466e+30)

GPSVN = 0
GPSPOS = 3
ZUPT = 6
CARMC = 9
GPSYAW = 12


def TauVal(x):
    if x > 3.402823466e+30 / 2:
        x = 0.0
    else:
        x = 1.0 / x


def ISZero(x):
    if -EPS < x[0, 0] < EPS and -EPS < x[1, 0] < EPS and -EPS < x[2, 0] < EPS:
        return True
    else:
        return False


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


class CIMU(object):
    nSamples = prefirst = 0
    phim = np.array([[0.0]] * 3)
    dvbm = np.array([[0.0]] * 3)
    wm_1 = np.array([[0.0]] * 3)
    vm_1 = np.array([[0.0]] * 3)

    def __init__(self):
        self.prefirst = 1

    def Update(self, wm, vm, nSamples):
        i = 0
        pcf = conefactors[nSamples - 2]
        cm = np.array([[0.0]] * 3)
        sm = np.array([[0.0]] * 3)
        wmm = np.array([[0.0]] * 3)
        vmm = np.array([[0.0]] * 3)

        self.nSamples = nSamples
        if nSamples == 1:
            if self.prefirst == 1:
                self.wm_1 = wm[0]
                self.vm_1 = vm[0]
                self.prefirst = 0
            cm = 1.0 / 12 * self.wm_1
            self.wm_1 = wm[0]
            sm = 1.0 / 12 * self.vm_1
            self.vm_1 = vm[0]
        if nSamples > 1:
            self.prefirst = 1
        for i in range(nSamples - 1):
            cm += pcf[i] * wm[i]
            sm += pcf[i] * vm[i]
            wmm += wm[i]
            vmm += vm[i]

        wmm += wm[i]
        vmm += vm[i]
        self.phim = wmm + cross(cm, wm[i])
        self.dvbm = vmm + cross(1.0 / 2 * wmm, vmm) + (cross(cm, vm[i]) + cross(sm, wm[i]))


class CSINS(object):
    nts = 0.0
    tk = 0.0
    eth = CEarth()
    imu = CIMU()
    qnb = np.array([[0.0]] * 4)
    Cnb = np.mat([[0.0] * 3] * 3)
    Cnb0 = np.mat([[0.0] * 3] * 3)
    Cbn = np.mat([[0.0] * 3] * 3)
    Kg = np.mat([[0.0] * 3] * 3)
    Ka = np.mat([[0.0] * 3] * 3)
    wib = np.array([[0.0]] * 3)
    fb = np.array([[0.0]] * 3)
    fn = np.array([[0.0]] * 3)
    an = np.array([[0.0]] * 3)
    web = np.array([[0.0]] * 3)
    wnb = np.array([[0.0]] * 3)
    att = np.array([[0.0]] * 3)
    vn = np.array([[0.0]] * 3)
    vb = np.array([[0.0]] * 3)
    pos = np.array([[0.0]] * 3)
    eb = np.array([[0.0]] * 3)
    db = np.array([[0.0]] * 3)
    ebMax = np.array([[1.0]] * 3)
    dbMax = np.array([[1.0]] * 3)

    tauGyro = np.array([[0.0]] * 3)
    tauAcc = np.array([[0.0]] * 3)
    Maa = np.mat([[0.0] * 3] * 3)
    Mav = np.mat([[0.0] * 3] * 3)
    Map = np.mat([[0.0] * 3] * 3)
    Mva = np.mat([[0.0] * 3] * 3)
    Mvv = np.mat([[0.0] * 3] * 3)
    Mvp = np.mat([[0.0] * 3] * 3)
    Mpv = np.mat([[0.0] * 3] * 3)
    Mpp = np.mat([[0.0] * 3] * 3)
    vnL = np.array([[0.0]] * 3)
    posL = np.array([[0.0]] * 3)
    CW = np.mat([[0.0] * 3] * 3)
    MpvCnb = np.mat([[0.0] * 3] * 3)

    def __init__(self, qnb0=np.array([[1.0], [0.0], [0.0], [0.0]]), vn0=np.array([[0.0]] * 3), pos0=np.array([[0.0]] * 3)):
        self.tk = 0.0
        self.qnb = qnb0
        self.vn = vn0
        self.pos = pos0
        self.eth.Update(pos0, vn0)
        self.Cnb = Q2M(self.qnb)
        self.att = M2A(self.Cnb)
        self.Cnb0 = self.Cnb
        self.Cbn = self.Cnb.T
        self.vb = self.Cbn * self.vn
        self.Kg = np.mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.Ka = np.mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.ebMax *= INF
        self.dbMax *= INF

    def etm(self):
        tl, secl, wN, wU, vE, vN = self.eth.tl, 1 / self.eth.cl, self.eth.wnie[1, 0], self.eth.wnie[2, 0], \
                                   self.vn[0, 0], self.vn[1, 0]
        secl2 = secl * secl
        f_RMh, f_RNh, f_clRNh = self.eth.f_RMh, self.eth.f_RNh, self.eth.f_clRNh
        f_RMh2, f_RNh2 = f_RMh * f_RMh, f_RNh * f_RNh
        Avn = askew(self.vn)
        Mp1 = np.mat([[0.0, 0.0, 0.0],
                      [-wU, 0.0, 0.0],
                      [wN, 0.0, 0.0]])
        Mp2 = np.mat([[0.0, 0.0, vN * f_RMh2],
                      [0.0, 0.0, -vE * f_RNh2],
                      [vE * secl2 * f_RNh, 0.0, -vE * tl * f_RNh2]])
        self.Maa = askew(-self.eth.wnin)
        self.Mav = np.mat([[0.0, -f_RMh, 0.0],
                           [f_RNh, 0.0, 0.0],
                           [tl * f_RNh, 0.0, 0.0]])
        self.Map = Mp1 + Mp2
        self.Mva = askew(self.fn)
        self.Mvv = Avn * self.Mav - askew(self.eth.wnie + self.eth.wnin)
        self.Mvp = Avn * (Mp1 + self.Map)
        scl = self.eth.sl * self.eth.cl
        self.Mvp[2, 0] = self.Mvp[2, 0] - glv.g0 * (5.27094e-3 * 2 * scl + 2.32718e-5 * 4 * self.eth.sl2 * scl)
        self.Mvp[2, 2] = self.Mvp[2, 2] + 3.086e-6
        self.Mpv = np.mat([[0.0, f_RMh, 0.0],
                           [f_clRNh, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
        self.Mpp = np.mat([[0.0, 0.0, -vN * f_RMh2],
                           [vE * tl * f_clRNh, 0.0, -vE * secl * f_RNh2],
                           [0.0, 0.0, 0.0]])

    def Update(self, wm, vm, nSamples, ts):
        self.nts = nSamples * ts
        self.tk += self.nts
        nts2 = self.nts / 2
        self.imu.Update(wm, vm, nSamples)
        self.imu.phim = self.Kg * self.imu.phim - self.eb * self.nts
        self.imu.dvbm = self.Ka * self.imu.dvbm - self.db * self.nts
        vn01 = self.vn + self.an * nts2
        pos01 = self.pos + self.eth.vn2pos(vn01, nts2)
        self.eth.Update(pos01, vn01)
        self.wib = self.imu.phim / self.nts
        self.fb = self.imu.dvbm / self.nts
        self.web = self.wib - self.Cbn * self.eth.wnie
        self.wnb = self.wib - QV_cross(Q_cross(self.qnb, rv2Q(self.imu.phim / 2)), self.eth.wnin)
        self.fn = QV_cross(self.qnb, self.fb)
        self.an = QV_cross(rv2Q(-self.eth.wnin * nts2), self.fn) + self.eth.gcc
        vn1 = self.vn + self.an * self.nts
        self.pos = self.pos + self.eth.vn2pos(self.vn + vn1, nts2)
        self.vn = vn1
        self.Cnb0 = self.Cnb
        self.qnb = Q_cross(Q_cross(rv2Q(-self.eth.wnin * self.nts), self.qnb), rv2Q(self.imu.phim))
        self.Cnb = Q2M(self.qnb)
        self.att = M2A(self.Cnb)
        self.Cbn = self.Cnb.T
        self.vb = self.Cbn * self.vn


class CAligni0(object):
    def __init__(self, pos=np.array([[0.0]] * 3)):
        self.tk = 0
        self.t0 = 10
        self.t1 = 10
        self.t2 = 0
        self.wmm = np.array([[0.0]] * 3)
        self.vmm = np.array([[0.0]] * 3)
        self.vib0 = np.array([[0.0]] * 3)
        self.vi0 = np.array([[0.0]] * 3)
        self.Pib01 = np.array([[0.0]] * 3)
        self.Pib02 = np.array([[0.0]] * 3)
        self.Pi01 = np.array([[0.0]] * 3)
        self.Pi02 = np.array([[0.0]] * 3)
        self.tmpPib0 = np.array([[0.0]] * 3)
        self.tmpPi0 = np.array([[0.0]] * 3)
        self.imu = CIMU()
        self.qib0b = np.array([[1.0], [0.0], [0.0], [0.0]])
        self.eth = CEarth()
        self.eth.Update(pos)

    def Update(self, wm, vm, nSamples, ts):
        nts = nSamples * ts
        self.imu.Update(wm, vm, nSamples)
        self.wmm = self.wmm + self.imu.phim
        self.vmm = self.vmm + self.imu.dvbm
        vtmp = QV_cross(self.qib0b, self.imu.dvbm)
        self.tk += nts
        Ci0n = pos2Cen(np.array([[self.eth.pos[0, 0]], [self.eth.wie * self.tk], [0.0]]))
        vtmp1 = Ci0n * (-self.eth.gn * nts)
        self.vib0 = self.vib0 + vtmp
        self.vi0 = self.vi0 + vtmp1
        self.Pib02 = self.Pib02 + self.vib0 * nts
        self.Pi02 = self.Pi02 + self.vi0 * nts
        self.t2 += 1
        if self.t2 > 3 * self.t0:
            self.t0 = self.t1
            self.Pib01 = self.tmpPib0
            self.Pi01 = self.tmpPi0
        elif self.t2 > 2 * self.t0 and self.t1 == self.t0:
            self.t1 = self.t2
            self.tmpPib0 = self.Pib02
            self.tmpPi0 = self.Pi02
        self.qib0b = Q_cross(self.qib0b, rv2Q(self.imu.phim))
        qnb = np.array([[0.0]] * 4)
        if self.t2 < 100:
            qnb[0, 0] = 1.0
        elif self.t2 < 1000:
            qnb = Mat2Qua(Vec2Mat(AlignCoarse(self.wmm, self.vmm, self.eth.pos[0, 0])))
        else:
            temp = Mat2Qua(Ci0n)
            temp[1, 0] *= -1
            temp[2, 0] *= -1
            temp[3, 0] *= -1
            qnb = Q_cross(Q_cross(temp, Mat2Qua(dv2att(self.Pi01, self.Pi02, self.Pib01, self.Pib02))), self.qib0b)
        return qnb


class CKalman(object):
    def __init__(self, nq, nr):
        self.tk = 0.0
        self.nq = nq
        self.nr = nr
        self.Ft = np.mat([[0.0] * nq] * nq)
        self.Pk = np.mat([[0.0] * nq] * nq)
        self.Hk = np.mat([[0.0] * nq] * nr)
        self.Qt = np.array([[0.0]] * nq)
        self.Pmin = np.array([[0.0]] * nq)
        self.Xk = np.array([[0.0]] * nq)
        self.Pmax = np.array([[INF]] * nq)
        self.Rk = np.array([[0.0]] * nr)
        self.Zk = np.array([[0.0]] * nr)
        self.measflag = 0

    def TimeUpdate(self, ts):
        self.tk += ts
        Fk = np.eye(self.nq) + self.Ft * ts
        self.Xk = Fk * self.Xk
        self.Pk = Fk * self.Pk * Fk.T
        self.Pk = self.Pk + self.Qt * ts

    def SetMeasFlag(self, flag):
        if flag == 0:
            self.measflag = 0
        else:
            self.measflag = self.measflag | flag

    def MeasUpdate(self, fading=1):
        for i in range(self.nr):
            if self.measflag & (0x01 << i):
                Hi = self.Hk[i]
                Pxz = self.Pk * Hi.T
                Pzz = (Hi * Pxz)[0, 0] + self.Rk[i, 0]
                Kk = Pxz * (1.0 / Pzz)
                self.Xk += Kk * (self.Zk[i, 0] - (Hi * self.Xk)[0, 0])
                self.Pk -= Kk * Pxz.T
        if fading > 1.0:
            self.Pk *= fading
        self.Pk = symmetry(self.Pk)

    def PKConstrain(self):
        i, nq1 = 0, self.nq + 1
        while i < 15:
            if self.Pmin[i, 0] > self.Pk[i, i] > EPS:
                self.Pk[i, i] = self.Pmin[i, 0]
            elif self.Pk[i, i] > self.Pmax[i, 0]:
                sqf = sqrt(self.Pmax[i, 0] / self.Pk[i, i]) * 0.5
                for k in range(self.nq):
                    self.Pk[i, k] *= sqf
                    self.Pk[k, i] *= sqf
            i += 1

    def Feedback(self, sins, tauphi, taudvn, taudpos, taueb, taudb):
        afa = 0.0
        hur10 = 36000
        if tauphi < hur10:
            if sins.nts < tauphi:
                afa = sins.nts / tauphi
            else:
                afa = 1.0
            phi = np.array([[self.Xk[0, 0]], [self.Xk[1, 0]], [self.Xk[2, 0]]])
            sins.qnb -= rv2Q(np.array([[phi[0, 0] * afa], [phi[1, 0] * afa], [phi[2, 0] * 0.1 * afa]]))
            self.Xk[0, 0] = phi[0, 0] * (1 - afa)
            self.Xk[1, 0] = phi[1, 0] * (1 - afa)
            self.Xk[2, 0] = phi[2, 0] * (1 - 0.1 * afa)
        if taudvn < hur10:
            if sins.nts < taudvn:
                afa = sins.nts / taudvn
            else:
                afa = 1.0
            dvn = np.array([[self.Xk[3, 0]], [self.Xk[4, 0]], [self.Xk[5, 0]]])
            sins.vn -= dvn * afa
            self.Xk[3, 0] = dvn[0, 0] * (1 - afa)
            self.Xk[4, 0] = dvn[1, 0] * (1 - afa)
            self.Xk[5, 0] = dvn[2, 0] * (1 - afa)
        if taudpos < hur10:
            if sins.nts < taudpos:
                afa = sins.nts / taudpos
            else:
                afa = 1.0
            dpos = np.array([[self.Xk[6, 0]], [self.Xk[7, 0]], [self.Xk[8, 0]]])
            sins.pos -= dpos * afa
            self.Xk[6, 0] = dpos[0, 0] * (1 - afa)
            self.Xk[7, 0] = dpos[1, 0] * (1 - afa)
            self.Xk[8, 0] = dpos[2, 0] * (1 - afa)
        if taueb < hur10:
            if sins.nts < taueb:
                afa = sins.nts / taueb
            else:
                afa = 1.0
            for i in range(3):
                if (sins.eb[i, 0] > sins.ebMax[i, 0] and self.Xk[i + 9, 0] < 0) or (
                        sins.eb[i, 0] < -sins.ebMax[i, 0] and self.Xk[i + 9, 0] > 0) or (
                        -sins.ebMax[i, 0] <= sins.eb[i, 0] <= sins.ebMax[i, 0]):
                    sins.eb[i, 0] += self.Xk[i + 9, 0] * afa
                    self.Xk[i + 9, 0] = self.Xk[i + 9, 0] * (1 - afa)
        if taudb < hur10:
            if sins.nts < taudb:
                afa = sins.nts / taudb
            else:
                afa = 1.0
            for i in range(3):
                if (sins.db[i, 0] > sins.dbMax[i, 0] and self.Xk[i + 12, 0] < 0) or (
                        sins.db[i, 0] < -sins.dbMax[i, 0] and self.Xk[i + 12, 0] > 0) or (
                        -sins.dbMax[i, 0] <= sins.db[i, 0] <= sins.dbMax[i, 0]):
                    sins.db[i, 0] += self.Xk[i + 12, 0] * afa
                    self.Xk[i + 12, 0] = self.Xk[i + 12, 0] * (1 - afa)

    def SetFt(self, sins):
        sins.etm()
        self.Ft[0, 1] = sins.Maa[0, 1]
        self.Ft[0, 2] = sins.Maa[0, 2]
        self.Ft[0, 4] = sins.Mav[0, 1]
        self.Ft[1, 0] = sins.Maa[1, 0]
        self.Ft[1, 2] = sins.Maa[1, 2]
        self.Ft[1, 3] = sins.Mav[1, 0]
        self.Ft[2, 0] = sins.Maa[2, 0]
        self.Ft[2, 1] = sins.Maa[2, 1]
        self.Ft[2, 3] = sins.Mav[2, 0]
        self.Ft[0, 8] = sins.Map[0, 2]
        self.Ft[0, 9] = -sins.Cnb[0, 0]
        self.Ft[0, 10] = -sins.Cnb[0, 1]
        self.Ft[0, 11] = -sins.Cnb[0, 2]
        self.Ft[1, 6] = sins.Map[1, 0]
        self.Ft[1, 8] = sins.Map[1, 2]
        self.Ft[1, 9] = -sins.Cnb[1, 0]
        self.Ft[1, 10] = -sins.Cnb[1, 1]
        self.Ft[1, 11] = -sins.Cnb[1, 2]
        self.Ft[2, 6] = sins.Map[2, 0]
        self.Ft[2, 8] = sins.Map[2, 2]
        self.Ft[2, 9] = -sins.Cnb[2, 0]
        self.Ft[2, 10] = -sins.Cnb[2, 1]
        self.Ft[2, 11] = -sins.Cnb[2, 2]
        self.Ft[3, 1] = sins.Mva[0, 1]
        self.Ft[3, 2] = sins.Mva[0, 2]
        self.Ft[3, 3] = sins.Mvv[0, 0]
        self.Ft[3, 4] = sins.Mvv[0, 1]
        self.Ft[3, 5] = sins.Mvv[0, 2]
        self.Ft[4, 0] = sins.Mva[1, 0]
        self.Ft[4, 2] = sins.Mva[1, 2]
        self.Ft[4, 3] = sins.Mvv[1, 0]
        self.Ft[4, 4] = sins.Mvv[1, 1]
        self.Ft[4, 5] = sins.Mvv[1, 2]
        self.Ft[5, 0] = sins.Mva[2, 0]
        self.Ft[5, 1] = sins.Mva[2, 1]
        self.Ft[5, 3] = sins.Mvv[2, 0]
        self.Ft[5, 4] = sins.Mvv[2, 1]
        self.Ft[3, 6] = sins.Mvp[0, 0]
        self.Ft[3, 8] = sins.Mvp[0, 2]
        self.Ft[3, 12] = sins.Cnb[0, 0]
        self.Ft[3, 13] = sins.Cnb[0, 1]
        self.Ft[3, 14] = sins.Cnb[0, 2]
        self.Ft[4, 6] = sins.Mvp[1, 0]
        self.Ft[4, 8] = sins.Mvp[1, 2]
        self.Ft[4, 12] = sins.Cnb[1, 0]
        self.Ft[4, 13] = sins.Cnb[1, 1]
        self.Ft[4, 14] = sins.Cnb[1, 2]
        self.Ft[5, 6] = sins.Mvp[2, 0]
        self.Ft[5, 8] = sins.Mvp[2, 2]
        self.Ft[5, 12] = sins.Cnb[2, 0]
        self.Ft[5, 13] = sins.Cnb[2, 1]
        self.Ft[5, 14] = sins.Cnb[2, 2]
        self.Ft[6, 4] = sins.Mpv[0, 1]
        self.Ft[6, 8] = sins.Mpp[0, 2]
        self.Ft[7, 3] = sins.Mpv[1, 0]
        self.Ft[7, 6] = sins.Mpp[1, 0]
        self.Ft[7, 8] = sins.Mpp[1, 2]
        self.Ft[8, 5] = sins.Mpv[2, 2]
        self.Ft[2, 15] = -sins.wib[2, 0] * sins.Cnb[2, 2]

    def SetHk(self):
        self.Hk[0, 3] = 1.0
        self.Hk[1, 4] = 1.0
        self.Hk[2, 5] = 1.0
        self.Hk[3, 6] = 1.0
        self.Hk[4, 7] = 1.0
        self.Hk[5, 8] = 1.0


class CTDKF(CKalman):
    def __init__(self, nq0, nr0, ts):
        super(CTDKF, self).__init__(nq=nq0, nr=nr0)

        # CTDKF
        self.iter = -2
        self.ifn = 0
        self.nts = 0.0
        self.Fk = np.mat([[0.0] * self.nq] * self.nq)
        self.Pk1 = np.mat([[0.0] * self.nq] * self.nq)
        self.Pxz = np.array([[0.0]] * self.nr)
        self.Qk = np.array([[0.0]] * self.nr)
        self.Kk = np.array([[0.0]] * self.nr)
        self.Hi = np.array([[0.0]] * self.nr)
        self.tmeas = np.array([[0.0]] * self.nr)
        self.meanfn = np.array([[0.0]] * 3)

        # CCarAHRS
        self.levelAlignOK = False
        self.yawAlignOK = False
        self.initPosOK = False
        self.measGPSVn = np.array([[0.0]] * 3)
        self.measGPSPos = np.array([[0.0]] * 3)
        self.measINSvn = np.array([[0.0]] * 3)
        self.measCMvb = np.array([[0.0]] * 3)
        self.measMag = np.array([[0.0]] * 3)
        self.measGPSYaw = 0
        self.align = CAligni0()
        self.sins = CSINS()

        sts = sqrt(ts)

        self.Pmax = np.square([
            10.0 * glv.deg, 10.0 * glv.deg, 30.0 * glv.deg, 50.0, 50.0, 50.0, 1.0e4 / glv.Re, 1.0e4 / glv.Re, 1.0e4,
            1000.0 * glv.dph, 1000.0 * glv.dph, 1000.0 * glv.dph, 100.0 * glv.mg, 100.0 * glv.mg, 100.0 * glv.mg,
            10000.0 * glv.ppm
        ]).reshape(-1, 1)
        self.Pmin = np.square([
            1.0 * glv.min, 1.0 * glv.min, 10.0 * glv.min, 0.01, 0.01, 0.1, 1.0 / glv.Re, 1.0 / glv.Re, 0.1,
            0.1 * glv.dph, 0.1 * glv.dph, 0.1 * glv.dph, 0.1 * glv.mg, 0.1 * glv.mg, 0.1 * glv.mg, 10 * glv.ppm
        ]).reshape(-1, 1)
        self.Pk = np.mat(
            np.diagflat(
                np.square(
                    [10.0 * glv.deg, 10.0 * glv.deg, 10.0 * glv.deg, 1.0, 1.0, 1.0, 100.0 / glv.Re, 100.0 / glv.Re,
                     100.0, 100.0 * glv.dph, 100.0 * glv.dph, 100.0 * glv.dph, 10.0 * glv.mg, 10.0 * glv.mg,
                     10.0 * glv.mg, 1000 * glv.ppm
                     ]
                )
            )
        )
        self.Qt = np.square([
            1.1 * glv.dpsh, 1.1 * glv.dpsh, 1.1 * glv.dpsh, 500.0 * glv.ugpsHz, 500.0 * glv.ugpsHz, 500.0 * glv.ugpsHz,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0 * glv.ugpsh, 100.0 * glv.ugpsh, 100.0 * glv.ugpsh, 100.0 * glv.ppmpsh
        ]).reshape(-1, 1)
        self.Rk = np.square([
            0.5, 0.5, 0.5, 10.0 / glv.Re, 10.0 / glv.Re, 10.0, 0.1 / sts, 0.1 / sts,
                           0.1 / sts, 10.0 / sts, 1000.0 / sts, 1.0 / sts, 1.0 * glv.deg
        ]).reshape(-1, 1)
        self.SetHk()
        self.Hk[ZUPT, 3] = 1.0
        self.Hk[ZUPT + 1, 4] = 1.0
        self.Hk[ZUPT + 2, 5] = 1.0
        self.Hk[GPSYAW, 2] = -1.0

    def TDUpdate(self, sins, ts, nStep):
        measRes = 0
        if nStep <= 0:
            nStep = 2 * (self.nq + self.nr) + 3
        self.nts += ts
        self.tk += ts
        self.meanfn = self.meanfn + sins.fn
        self.ifn += 1
        for i in range(nStep):
            if self.iter == -2:
                vtmp = self.meanfn * (1.0 / self.ifn)
                self.meanfn = np.array([[0.0]] * 3)
                self.ifn = 0
                sins.fn = vtmp
                self.SetFt(sins)
                sins.fn = vtmp
                self.MeasRearrange(sins)
            elif self.iter == -1:
                self.Fk = np.eye(self.nq) + self.Ft * self.nts
                self.Qk = self.Qt * self.nts
                self.Xk = self.Fk * self.Xk
                self.nts = 0.0
            elif self.iter < self.nq:
                row = self.iter
                for i in range(self.nq):
                    f = 0.0
                    for j in range(self.nq):
                        f += self.Fk[row, j] * self.Pk[j, i]
                    self.Pk1[row, i] = f
            elif self.iter < self.nq * 2:
                row = self.iter - self.nq
                for i in range(self.nq):
                    f = 0.0
                    for j in range(self.nq):
                        f += self.Pk1[row, j] * self.Fk.T[j, i]
                    self.Pk[row, i] = f
                if row == self.nq - 1:
                    self.Pk += self.Qk
            elif self.iter < 2 * (self.nq + self.nr):
                row = int((self.iter - 2 * self.Ft.shape[0]) / 2)
                flag = self.measflag & (0x01 << row)
                if flag:
                    measRes |= flag
                    if (self.iter - 2 * self.Ft.shape[0]) % 2 == 0:
                        self.Hi = self.Hk[row]
                        self.Pxz = self.Pk * self.Hi.T
                        Pzz = (self.Hi * self.Pxz)[0, 0] + self.Rk[row, 0]
                        self.Kk = self.Pxz * 1.0 / Pzz
                    else:
                        self.Xk += self.Kk * (self.Zk[row, 0] - (self.Hi * self.Xk)[0, 0])
                        self.Pk -= self.Kk * self.Pxz.T
            elif self.iter >= 2 * (self.nq + self.nr):
                self.PKConstrain()
                self.Pk = symmetry(self.Pk)
                self.SetMeasFlag(0)
                self.iter = -3
            self.iter += 1
        self.Feedback(sins, 10.0, 1.0, 1.0, 100.0, 100.0)

    def SetMeasGPSVn(self, vnGPS):
        self.measGPSVn = vnGPS
        self.tmeas[GPSVN, 0] = self.tk

    def SetMeasGPSPos(self, posGps):
        self.measGPSPos = posGps
        self.tmeas[GPSPOS, 0] = self.tk
        if not self.initPosOK:
            self.sins.pos = posGps
            self.initPosOK = True

    def SetMeasZUPT(self):
        self.measINSvn = self.sins.vn
        self.tmeas[ZUPT, 0] = self.tk

    def SetMeasMag(self, mag):
        self.measMag = mag

    def SetMeasGPSYaw(self, gpsYaw):
        self.measGPSYaw = gpsYaw
        self.tmeas[GPSYAW, 0] = self.tk

    def SetMeasMC(self):
        self.Hk[CARMC + 0, 3] = self.sins.Cnb[0, 0]
        self.Hk[CARMC + 0, 4] = self.sins.Cnb[1, 0]
        self.Hk[CARMC + 0, 5] = self.sins.Cnb[2, 0]
        self.Hk[CARMC + 1, 3] = self.sins.Cnb[0, 1]
        self.Hk[CARMC + 1, 4] = self.sins.Cnb[1, 1]
        self.Hk[CARMC + 1, 5] = self.sins.Cnb[2, 1]
        self.Hk[CARMC + 2, 3] = self.sins.Cnb[0, 2]
        self.Hk[CARMC + 2, 4] = self.sins.Cnb[1, 2]
        self.Hk[CARMC + 2, 5] = self.sins.Cnb[2, 2]
        self.measCMvb = self.sins.vb
        if self.measCMvb[1, 0] > 20:
            self.measCMvb[1, 0] -= 20
        elif self.measCMvb[1, 0] < -1:
            self.measCMvb[1, 0] -= -1
        self.tmeas[CARMC, 0] = self.tk

    def MeasRearrange(self, sins):
        if self.yawAlignOK and self.measGPSVn[2, 0] != 0 and self.tk - self.tmeas[GPSVN, 0] < 0.5:
            for i in range(3):
                self.Zk[GPSVN + i, 0] = self.sins.vn[i, 0] - self.measGPSVn[i, 0]
            self.measGPSVn = np.array([[0.0]] * 3)
            self.SetMeasFlag(0x07)

        if self.yawAlignOK and self.measGPSPos[2, 0] != 0 and self.tk - self.tmeas[GPSPOS, 0] < 0.5:
            temp = self.sins.eth.vn2pos(self.vn, self.tk - self.tmeas[GPSPOS, 0] - self.sins.nts)
            for i in range(3):
                self.Zk[GPSPOS + i, 0] = self.sins.pos[i, 0] - self.measGPSPos[i, 0] - temp[i, 0]
            self.measGPSPos = np.array([[0.0]] * 3)
            self.SetMeasFlag(0x38)
        if self.levelAlignOK and self.measINSvn[2, 0] != 0 and self.tk - self.tmeas[ZUPT, 0] < 0.1:
            for i in range(3):
                self.Zk[ZUPT + i, 0] = self.measINSvn[i, 0] - self.sins.an[i, 0] * (
                        self.tk - self.tmeas[ZUPT, 0] - self.sins.nts)
            self.measINSvn = np.array([[0.0]] * 3)
            self.SetMeasFlag(0x01C0)
        elif self.levelAlignOK and self.measCMvb[2, 0] != 0 and self.tk - self.tmeas[CARMC, 0] < 0.1 and self.tk - \
                self.tmeas[GPSVN, 0] > 10.0:
            for i in range(3):
                self.Zk[CARMC + i, 0] = self.measCMvb[i, 0]
                self.measCMvb = np.array([[0.0]] * 3)
                self.SetMeasFlag(0x0E00)
        if self.yawAlignOK and self.measGPSYaw != 0 and self.tk - self.tmeas[GPSYAW, 0] < 0.5:
            self.Zk[GPSYAW, 0] = self.sins.att[2, 0] - self.measGPSYaw + 2 * glv.deg
            self.measGPSYaw = 0

    def Update(self, wm, vm, ts):
        res = 0
        if not self.levelAlignOK:
            self.sins.qnb = self.align.Update(wm, vm, 1, ts)
            if self.align.tk > 5:
                self.levelAlignOK = True
        if self.levelAlignOK and not self.yawAlignOK:
            if sqrt(self.measGPSVn[0, 0] ** 2 + self.measGPSVn[1, 0] ** 2) > 2.0:
                att = M2A(Q2M(self.sins.qnb))
                att[2, 0] = atan2(-self.measGPSVn[0, 0], self.measGPSVn[1, 0])
                self.sins.qnb = Mat2Qua(Vec2Mat(att))
                self.yawAlignOK = True
                self.sins.tk = self.tk
            if self.measGPSYaw != 0:
                att = M2A(Q2M(self.sins.qnb))
                att[2, 0] = self.measGPSYaw
                self.sins.qnb = Mat2Qua(Vec2Mat(att))
                self.yawAlignOK = True
                self.sins.tk = self.tk

        self.sins.Update(wm, vm, 1, ts)
        self.TDUpdate(self.sins, ts, 20)


class CRAvar:
    def __init__(self, nR0, ts):
        self.nR0 = nR0
        self.ts = ts
        self.Rmaxflag = [0] * nR0
        self.R0 = [0.0] * nR0
        self.Rmax = [0.0] * nR0
        self.Rmin = [0.0] * nR0
        self.tstau = [0.0] * nR0
        self.r0 = [0.0] * nR0

    def setR0(self, r0):
        for i in range(self.nR0):
            self.R0[i] = r0[i] * r0[i]
            self.Rmax[i] = 100.0 * self.R0[i]
            self.Rmin[i] = 0.01 * self.R0[i]
            self.r0[i] = 0.0
            self.Rmaxflag[i] = 1

    def setTau(self, tau):
        for i in range(self.nR0):
            if self.ts > tau[i]:
                self.tstau[i] = 1.0
            else:
                self.tstau[i] = self.ts / tau[i]

    def setRmax(self, rmax):
        for i in range(self.nR0):
            rmax2 = rmax[i] * rmax[i]
            if rmax2 > self.R0[i]:
                self.Rmax[i] = rmax2

    def setRmin(self, rmin):
        for i in range(self.nR0):
            rmin2 = rmin[i] * rmin[i]
            if EPS < rmin2 < self.R0[i]:
                self.Rmin[i] = rmin2

    def Update(self, r):
        for i in range(self.nR0):
            dr2 = r[i] - self.r0[i]
            dr2 = dr2 * dr2
            self.r0[i] = r[i]
            if dr2 > self.R0[i]:
                self.R0[i] = dr2
            else:
                self.R0[i] = (1.0 - self.tstau[i]) * self.R0[i] + self.tstau[i] * dr2

            if self.R0[i] < self.Rmin[i]:
                self.R0[i] = self.Rmin[i]
            if self.R0[i] > self.Rmax[i]:
                self.R0[i] = self.Rmax[i]
                self.Rmaxflag[i] = 1
            else:
                self.Rmaxflag[i] = 0

    def getindex(self, k):
        if self.Rmaxflag[k] == 1:
            return INF
        else:
            return sqrt(self.R0[k])
