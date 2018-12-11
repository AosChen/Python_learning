import numpy as np
import math

from VDR.Kalman_Filter.EKF import EKF
from VDR.Kalman_Filter.Smooth import Smooth

from VDR.Kalman_Filter.KF_NN import *

rad2deg = 57.295780490442968321226628812406
deg2rad = 0.01745329237161968996669562749648

earthRe, earthr, earthf, earthe, earthwie = 6378137, 6356752.3142, 1 / 298.257, 0.0818, 7.292e-5


def getRotationMatrix(gravity, geomagnetic):
    Ax, Ay, Az = gravity[0], gravity[1], gravity[2]
    normsqA = Ax * Ax + Ay * Ay + Az * Az
    g = 9.81
    freeFallGravitySquared = 0.01 * g * g
    if normsqA < freeFallGravitySquared:
        return None

    Ex, Ey, Ez = geomagnetic[0], geomagnetic[1], geomagnetic[2]
    Hx, Hy, Hz = Ey * Az - Ez * Ay, Ez * Ax - Ex * Az, Ex * Ay - Ey * Ax
    normH = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if normH < 0.1:
        return None
    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz *= invH
    invA = 1.0 / math.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA
    Az *= invA
    Mx, My, Mz = Ay * Hz - Az * Hy, Az * Hx - Ax * Hz, Ax * Hy - Ay * Hx
    R = [None] * 9
    R[0] = Hx
    R[1] = Hy
    R[2] = Hz
    R[3] = Mx
    R[4] = My
    R[5] = Mz
    R[6] = Ax
    R[7] = Ay
    R[8] = Az
    return np.mat(R).reshape(3, 3)


class Calc(object):
    accCalStc_X, accCalStc_Y, accCalStc_Z = -0.09092, 0.081208, 0.015632
    gyoCalStc_X, gyoCalStc_Y, gyoCalStc_Z = 0.00122173049021512, 0.00122173049021512, 0.00122173049021512

    firstGPSOff = 0
    GPSOff = 0

    lastGPSh = 0

    Vccq = [None] * 3

    lastVx, lastVy, lastVz = 0.0, 0.0, 0.0
    GPSVe, GPSVn, GPSVu = 0.0, 0.0, 0.0
    last_L, last_E, last_h = 0.0, 0.0, 0.0
    L, E, h = 0.0, 0.0, 0.0
    Rm, Rn, R0 = 0.0, 0.0, 0.0

    tao = 1

    mahonyR = np.mat([[0.0] * 3] * 3)
    Fn = np.mat([0.0] * 3).reshape(3, 1)
    LineFn = np.mat([0.0] * 3).reshape(3, 1)

    r = [0.0] * 9
    Yaw = 0.0

    def __init__(self, samplePeriod=0.01, smooth_size=12):
        self.samplePeriod = samplePeriod
        self.kf = EKF()
        self.smooth = Smooth(smooth_size)
        self.KF_NN = KalmanFilter_NN()

    def deal_data(self, data_set, do_NN=False):
        start_Yaw, start_Vx, start_Vy, start_E, start_L = 0, 0, 0, 0, 0
        for (datas, i) in zip(data_set, range(len(data_set))):
            ax, ay, az = -(datas[0] - self.accCalStc_X), -(datas[1] - self.accCalStc_Y), -(datas[2] - self.accCalStc_Z)
            lax, lay, laz = -datas[3], -datas[4], -datas[5]
            gx, gy, gz = -datas[6], -datas[7], -datas[8]
            gyox, gyoy, gyoz = (datas[9] - self.gyoCalStc_X) * rad2deg, (datas[10] - self.gyoCalStc_Y) * rad2deg, \
                               (datas[11] - self.gyoCalStc_Z) * rad2deg
            mx, my, mz = datas[12], datas[13], datas[14]

            GPSLongitude, GPSLatitude, GPSHeight, GPSv, GPSYaw = datas[-9] * deg2rad, datas[-8] * deg2rad, datas[-7], \
                                                                 datas[-6], datas[-5]
            GPSHDOP = datas[-2]

            if self.firstGPSOff == 0:
                if GPSHDOP > 4:
                    return 'Please wait until GPS is working.'
                elif GPSLatitude != 0:
                    self.last_E, self.last_L, self.last_h = GPSLongitude, GPSLatitude, GPSHeight
                    self.lastVx = GPSv * math.sin(GPSYaw * deg2rad)
                    self.lastVy = GPSv * math.cos(GPSYaw * deg2rad)
                    self.lastVz = 0.0
                    self.E, self.L, self.h = self.last_E, self.last_L, self.last_h
                    self.firstGPSOff = 1
                    self.mahonyR = getRotationMatrix([gx, gy, gz], [mx, my, mz])
                    self.Yaw = GPSYaw
            else:
                data = [ax, ay, az, lax, lay, laz, gx, gy, gz, gyox, gyoy, gyoz, mx, my, mz]
                data = self.smooth.getSmoothResult(data)

                self.mahonyR = getRotationMatrix([data[6], data[7], data[8]], [data[12], data[13], data[14]])
                GG = math.sqrt(data[6] * data[6] + data[7] * data[7] + data[8] * data[8])
                ag = np.dot(np.array([data[9], data[10], data[11]]), np.array([data[6], data[7], data[8]]) / GG)

                self.Yaw += (ag * self.samplePeriod)
                if self.Yaw > 360:
                    self.Yaw -= 360
                if self.Yaw < 0:
                    self.Yaw == 360
                self.Rm = earthRe * (1 - 2 * earthf + 3 * earthf * math.sin(self.last_L) * math.sin(self.last_L))
                self.Rn = earthRe * (1 + earthf * math.sin(self.last_L) * math.sin(self.last_L))

                self.Fn = self.mahonyR * np.mat([data[0], data[1], data[2]]).reshape(3, 1)
                self.LineFn = self.mahonyR * np.mat([data[3], data[4], data[5]]).reshape(3, 1)

                self.lastVx += self.LineFn[0, 0] * self.samplePeriod
                self.lastVy += self.LineFn[1, 0] * self.samplePeriod
                self.lastVz += self.LineFn[2, 0] * self.samplePeriod

                V_mod = math.sqrt(self.lastVx * self.lastVx + self.lastVy * self.lastVy)
                self.lastVx = V_mod * math.sin(self.Yaw * deg2rad)
                self.lastVy = V_mod * math.cos(self.Yaw * deg2rad)

                self.L += (self.lastVy / self.Rm) * self.samplePeriod
                self.E += (self.lastVx / (math.cos(self.last_L) * self.Rn)) * self.samplePeriod
                self.h += self.lastVz * self.samplePeriod

                isUsed = True
                if GPSv == 0 and GPSYaw == 0:
                    isUsed = False

                if i == 0:
                    start_Yaw = self.Yaw
                    start_Vx = self.lastVx
                    start_Vy = self.lastVy
                    start_E = self.E
                    start_L = self.L

                INPUT = [self.lastVx - start_Vx, self.lastVy - start_Vy,
                         self.E - start_E, self.L - start_L,
                         abs(self.Yaw - start_Yaw) * deg2rad, self.Yaw * deg2rad]

                if i == len(data_set) - 1:
                    if not do_NN and GPSHDOP < 3.0 and isUsed:
                        self.GPSVe = GPSv * math.sin(GPSYaw * deg2rad)
                        self.GPSVn = GPSv * math.cos(GPSYaw * deg2rad)
                        self.GPSVu = GPSHeight - self.lastGPSh
                        if self.GPSOff == 1:
                            self.E = GPSLongitude
                            self.L = GPSLatitude
                            self.h = GPSHeight
                            self.GPSOff = 0

                        Dpv = np.mat([
                            self.L - GPSLatitude, self.E - GPSLongitude, self.h - GPSHeight,
                            self.lastVx - self.GPSVe, self.lastVy - self.GPSVn, self.lastVz - self.GPSVu]).reshape(-1,
                                                                                                                   1)

                        XX = self.kf.kalman_GPS_INS_pv(Dpv, self.lastVx, self.lastVy, self.lastVz,
                                                       self.L, self.h, self.mahonyR,
                                                       self.Fn, self.tao, self.Rm, self.Rn)

                        self.lastVx -= XX[3, 0]
                        self.lastVy -= XX[4, 0]
                        self.lastVz -= XX[5, 0]
                        self.L -= XX[6, 0]
                        self.E -= XX[7, 0]
                        self.h -= XX[8, 0]

                        self.gyoCalStc_X = XX[9, 0] - XX[12, 0] / 300
                        self.gyoCalStc_Y = XX[10, 0] - XX[13, 0] / 300
                        self.gyoCalStc_Z = XX[11, 0] - XX[14, 0] / 300
                        self.accCalStc_X = -XX[15, 0] / 1000
                        self.accCalStc_Y = -XX[16, 0] / 1000
                        self.accCalStc_Z = -XX[17, 0] / 1000

                        self.lastGPSh = GPSHeight

                        Lat_delta, Lot_delta = self.L - GPSLatitude, self.E - GPSLongitude
                        distance = 2 * math.asin(math.sqrt(pow(math.sin(Lat_delta / 2), 2) + math.cos(self.L) *
                                                           math.cos(GPSLatitude) * pow(math.sin(Lot_delta / 2),
                                                                                       2))) * earthRe
                        TARGET = [-XX[3, 0], -XX[4, 0], -XX[7, 0], -XX[6, 0], (GPSYaw - self.Yaw) * deg2rad,
                                  GPSYaw * deg2rad]
                        if distance >= 20:
                            TARGET = [self.lastVx - self.GPSVe, self.lastVy - self.GPSVn, self.E - GPSLongitude,
                                      self.L - GPSLatitude, (GPSYaw - self.Yaw) * deg2rad, GPSYaw * deg2rad]
                            self.E, self.L, self.h, self.Yaw = GPSLongitude, GPSLatitude, GPSHeight, GPSYaw
                            self.lastVx, self.lastVy, self.lastVz = self.GPSVe, self.GPSVn, self.GPSVu
                        # INPUT = torch.tensor(INPUT)
                        # TARGET = torch.tensor(TARGET)
                        # loss_function = nn.MSELoss()
                        # optimizier = optim.Adam(self.KF_NN.parameters(), lr=0.01)
                        # OUTPUT = self.KF_NN(INPUT)
                        # loss = loss_function(OUTPUT, TARGET)
                        # optimizier.zero_grad()
                        # loss.backward()
                        # print('loss is ' + str(loss))
                        # optimizier.step()
                    elif do_NN:
                        XX = self.kf.X
                        self.lastVx -= XX[3, 0]
                        self.lastVy -= XX[4, 0]
                        self.lastVz -= XX[5, 0]
                        self.L -= XX[6, 0]
                        self.E -= XX[7, 0]
                        self.h -= XX[8, 0]
                        # pass
                        # output = self.KF_NN(INPUT)
                        # self.lastVx -= output[0].item()
                        # self.lastVy -= output[1].item()
                        # self.L -= output[3].item()
                        # self.E -= output[2].item()
                        # self.Yaw = output[5].item() * rad2deg
                    if GPSHDOP >= 3.0:
                        self.GPSOff = 1

                self.last_L = self.L
                self.last_h = self.h
        # XX = self.kf.X
        # self.lastVx -= XX[3, 0]
        # self.lastVy -= XX[4, 0]
        # self.lastVz -= XX[5, 0]
        # self.L -= XX[6, 0]
        # self.E -= XX[7, 0]
        # self.h -= XX[8, 0]
        return [self.E * rad2deg, self.L * rad2deg, self.h]
