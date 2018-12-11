import math


def invSqrt(x):
    return 1 / math.sqrt(x)


class MahonyAHRS(object):

    def __init__(self, sampleFreq):
        self.__twoKp = 0.001
        self.__twoKi = 0.001
        self.__integralFBX, self.__integralFBy, self.__integralFBz = 0.0, 0.0, 0.0
        self.__q0, self.__q1, self.__q2, self.__q3 = 1.0, 0.0, 0.0, 0.0
        self.__sampleFreq = sampleFreq

    def MahonyAHRSupdateIMU(self, gx, gy, gz, ax, ay, az):
        recipNorm = 0.0
        qa, qb, qc = 0.0, 0.0, 0.0

        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            recipNorm = invSqrt(ax * ax + ay * ay + az * az)
            ax *= recipNorm
            ay *= recipNorm
            az *= recipNorm

            vx = 2 * (self.__q1 * self.__q3 - self.__q0 * self.__q2)
            vy = 2 * (self.__q0 * self.__q1 + self.__q2 * self.__q3)
            vz = 1 - 2 * (self.__q1 * self.__q1 + self.__q2 * self.__q2)

            ex = (ay * vz - az * vy)
            ey = (az * vx - ax * vz)
            ez = (ax * vy - ay * vx)

            if self.__twoKi > 0.0:
                self.__integralFBX += self.__twoKi * ex
                self.__integralFBy += self.__twoKi * ey
                self.__integralFBz += self.__twoKi * ez
                gx += self.__integralFBX
                gy += self.__integralFBy
                gz += self.__integralFBz
            else:
                self.__integralFBX = 0.0
                self.__integralFBy = 0.0
                self.__integralFBz = 0.0

            gx += self.__twoKp * ex
            gy += self.__twoKp * ey
            gz += self.__twoKp * ez

        qa = self.__q0
        qb = self.__q1
        qc = self.__q2
        self.__q0 += (-qb * gx - qc * gy - self.__q3 * gz) * 0.5 / self.__sampleFreq
        self.__q1 += (qa * gx + qc * gz - self.__q3 * gy) * 0.5 / self.__sampleFreq
        self.__q2 += (qa * gy - qb * gz + self.__q3 * gx) * 0.5 / self.__sampleFreq
        self.__q3 += (qa * gz + qb * gy - qc * gx) * 0.5 / self.__sampleFreq

        recipNorm = invSqrt(
            self.__q0 * self.__q0 + self.__q1 * self.__q1 + self.__q2 * self.__q2 + self.__q3 * self.__q3)
        self.__q0 *= recipNorm
        self.__q1 *= recipNorm
        self.__q2 *= recipNorm
        self.__q3 *= recipNorm

    def getQ(self):
        return [self.__q0, self.__q1, self.__q2, self.__q3]

    def setQ(self, Q):
        self.__q0 = Q[0]
        self.__q1 = Q[1]
        self.__q2 = Q[2]
        self.__q3 = Q[3]
