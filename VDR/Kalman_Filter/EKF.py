import numpy as np
import math

deg2rad = 0.01745329237161968996669562749648
rad2deg = 57.295780490442968321226628812406
G0 = 9.8015
wie = 7.292e-5
Q_wg = (0.5 * deg2rad / 3600) ** 2
Q_wr = (0.1 * deg2rad / 3600) ** 2
Q_wa = (0.5e-4 * G0) ** 2
Tg = 300
Ta = 1000
Rlamt, Rl, Rh, Rvx, Rvy, Rvz = 1e-5 * deg2rad / 60, 1e-5 * deg2rad / 60, \
                               1e-5 * deg2rad / 60, 1e-7, 1e-7, 5e-9
Q = np.diagflat([Q_wg] * 3 + [Q_wr] * 3 + [Q_wa] * 3)
R = np.diagflat([Rlamt, Rl, Rh, Rvx, Rvy, Rvz])
PP0k = [(1 / (36 * 57)) ** 2, (1 / (36 * 57)) ** 2, (1 / (36 * 57)) ** 2,
        1e-4, 1e-4, 1e-4, 0, 0, 1,
        (0.1 * deg2rad / 3600) ** 2, (0.1 * deg2rad / 3600) ** 2, (0.1 * deg2rad / 3600) ** 2,
        (0.04 * deg2rad / 3600) ** 2, (0.04 * deg2rad / 3600) ** 2, (0.04 * deg2rad / 3600) ** 2,
        1e-8, 1e-8, 1e-8]


class EKF(object):
    def __init__(self):
        self.PP = np.mat(np.diagflat(PP0k))
        self.X = np.mat([[0] for _ in range(18)])

    def kalman_GPS_INS_pv(self, Dpv, Ve, Vn, Vu, L, h, mahonyR, Fn, tao, Rm, Rn):
        fe, fn, fu = Fn[0, 0], Fn[1, 0], Fn[2, 0]
        C11, C12, C13 = mahonyR[0, 0], mahonyR[0, 1], mahonyR[0, 2]
        C21, C22, C23 = mahonyR[1, 0], mahonyR[1, 1], mahonyR[1, 2]
        C31, C32, C33 = mahonyR[2, 0], mahonyR[2, 1], mahonyR[2, 2]
        cosL, sinL, tanL = math.cos(L), math.sin(L), math.tan(L)
        secL = 1 / cosL
        secL2 = secL ** 2
        Rnh, Rmh = Rn + h, Rm + h
        Rnh2, Rmh2 = Rnh ** 2, Rmh ** 2

        F = np.mat(
            [
                [               0               ,wie * sinL + Ve * tanL / Rnh,-(wie * cosL + Ve / Rnh),                0                  ,            -1 / Rmh            ,              0              ,                            0                            ,        0        ,            Vn / Rmh2            , C11 , C12 , C13 ,  C11  ,  C12  ,  C13  ,   0   ,   0   ,   0   ],
                [-(wie * sinL + Ve * tanL / Rnh),             0              ,        -Vn / Rmh       ,            1 / Rnh                ,                0               ,              0              ,                        -wie * sinL                      ,        0        ,           -Ve / Rnh2            , C21 , C22 , C23 ,  C21  ,  C22  ,  C23  ,   0   ,   0   ,   0   ],
                [     wie * cosL + Ve / Rnh     ,           Vn / Rmh         ,            0           ,            tanL / Rnh             ,                0               ,              0              ,                wie * cosL + Ve * secL2 / Rnh            ,        0        ,         -Ve * tanL / Rnh2       , C31 , C32 , C33 ,  C31  ,  C32  ,  C33  ,   0   ,   0   ,   0   ],
                [               0               ,             -fu            ,            fn          ,        (Vn * tanL - Vu) / Rmh     ,2 * wie * sinL + Ve * tanL / Rnh,-(2 * wie * cosL + Ve / Rnh) ,2 * wie * (cosL * Vn + sinL * Vu) + Ve * Vn * secL2 / Rnh,        0        ,(Ve * Vu - Ve * Vn * tanL) / Rnh2,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,  C11  ,  C12  ,  C13  ],
                [               fu              ,             0              ,            -fe         ,-2 * (wie * sinL + Ve * tanL / Rnh),            -Vu / Rmh           ,         -Vn / Rmh           ,        -(2 * wie * cosL * Ve + Ve * Ve * secL2 / Rnh)   ,        0        ,(Ve * Ve * tanL + Vn * Vu) / Rnh2,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,  C21  ,  C22  ,  C23  ],
                [               -fn             ,             fe             ,            0           ,    2 * (wie * cosL + Ve / Rnh)    ,            2 * Vn / Rmh        ,              0              ,                    -2 * wie * sinL * Ve                 ,        0        ,   -(Ve * Ve + Vn * Vn) / Rnh2   ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,  C31  ,  C32  ,  C33  ],
                [               0               ,             0              ,            0           ,                0                  ,                1 / Rmh         ,              0              ,                            0                            ,        0        ,           -Vn / Rmh2            ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,            secL / Rnh             ,                0               ,              0              ,                    Ve * secL * tanL / Rnh               ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              1              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,-1 / Tg,   0   ,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,-1 / Tg,   0   ,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,-1 / Tg,   0   ,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,-1 / Ta,   0   ,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,-1 / Ta,   0   ],
                [               0               ,             0              ,            0           ,                0                  ,                0               ,              0              ,                            0                            ,        0        ,               0                 ,  0  ,  0  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,-1 / Ta]
            ]
        )
        G = np.array(
            [
                [ C11 , C12 , C13 ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [ C21 , C22 , C23 ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [ C31 , C32 , C33 ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  1  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  1  ,  0  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  1  ,  0  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  1  ,  0  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  1  ,  0  ],
                [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  1  ]
            ]

        )
        H = np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        eye18 = np.mat(np.eye(18))
        A = eye18+F*tao
        B = (eye18+(F*tao*0.5))*(G*tao)

        P = A*self.PP*A.T + B*Q*B.T
        K = P*H.T*(H*P*H.T+R).I
        self.PP = (eye18-(K*H))*P
        self.PP = (self.PP+self.PP.T)*0.5
        Z = Dpv
        self.X = A*self.X + K*(Z - H*A*self.X)
        return self.X