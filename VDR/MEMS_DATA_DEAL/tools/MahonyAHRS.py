import math


def invSqrt(x):
    return 1 / math.sqrt(x)


class MahonyAHRS(object):

    def __init__(self, sampleFreq):
        self.__twoKp = 2 * 2.146
        self.__twoKi = 2.146 * 2.146
        self.__integralFBX, self.__integralFBy, self.__integralFBz = 0.0, 0.0, 0.0
        self.__q0, self.__q1, self.__q2, self.__q3 = 1.0, 0.0, 0.0, 0.0
        self.__sampleFreq = sampleFreq

    def MahonyAHRSupdate(self, gx, gy, gz, ax, ay, az, mx, my, mz):
        recipNorm = 0.0
        qa, qb, qc = 0.0, 0.0, 0.0

        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            recipNorm = invSqrt(ax * ax + ay * ay + az * az)
            ax *= recipNorm
            ay *= recipNorm
            az *= recipNorm

            recipNorm = invSqrt(mx * mx + my * my + mz * mz)
            mx *= recipNorm
            my *= recipNorm
            mz *= recipNorm

            q0q0 = self.__q0 * self.__q0
            q0q1 = self.__q0 * self.__q1
            q0q2 = self.__q0 * self.__q2
            q0q3 = self.__q0 * self.__q3
            q1q1 = self.__q1 * self.__q1
            q1q2 = self.__q1 * self.__q2
            q1q3 = self.__q1 * self.__q3
            q2q2 = self.__q2 * self.__q2
            q2q3 = self.__q2 * self.__q3
            q3q3 = self.__q3 * self.__q3

            hx = 2.0 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
            hy = 2.0 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
            bx = math.sqrt(hx * hx + hy * hy)
            bz = 2.0 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))

            vx = q1q3 - q0q2
            vy = q0q1 + q2q3
            vz = q0q0 - 0.5 + q3q3

            wx = bx * (0.5 - q2q2 - q3q3) + bz * (q1q3 - q0q2)
            wy = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3)
            wz = bx * (q0q2 + q1q3) + bz * (0.5 - q1q1 - q2q2)

            ex = (ay * vz - az * vy) + (my * wz - mz * wy)
            ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
            ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

            if self.__twoKi > 0.0:
                self.__integralFBX += self.__twoKi * ex * (1 / self.__sampleFreq)
                self.__integralFBy += self.__twoKi * ey * (1 / self.__sampleFreq)
                self.__integralFBz += self.__twoKi * ez * (1 / self.__sampleFreq)
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

        gx *= 0.5 * (1.0 / self.__sampleFreq)
        gy *= 0.5 * (1.0 / self.__sampleFreq)
        gz *= 0.5 * (1.0 / self.__sampleFreq)
        qa = self.__q0
        qb = self.__q1
        qc = self.__q2
        self.__q0 += (-qb * gx - qc * gy - self.__q3 * gz)
        self.__q1 += (qa * gx + qc * gz - self.__q3 * gy)
        self.__q2 += (qa * gy - qb * gz + self.__q3 * gx)
        self.__q3 += (qa * gz + qb * gy - qc * gx)

        recipNorm = invSqrt(
            self.__q0 * self.__q0 + self.__q1 * self.__q1 + self.__q2 * self.__q2 + self.__q3 * self.__q3)
        self.__q0 *= recipNorm
        self.__q1 *= recipNorm
        self.__q2 *= recipNorm
        self.__q3 *= recipNorm

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
