import csv
from VDR.PSINS_2017.Algo_Class import *

DIR = r'D:\Project\VisualStudio\PSINS\psins_vs2017'
InputFile = DIR + r'\data1 - 副本.csv'
csv_reader = csv.reader(open(InputFile, 'r'))

kf_writer = csv.writer(open(DIR + r'\kf.csv', 'w', newline=''))
sins_writer = csv.writer(open(DIR + r'\sins.csv', 'w', newline=''))
rt_writer = csv.writer(open(DIR + r'\rt.csv', 'w', newline=''))

timu = 0

# 读取数据
index = 0
datas = []
for row in csv_reader:
    if index == 0:
        index += 1
        continue
    else:
        datas.append(list(map(float, row)))

pos0 = np.array([[34.2120468 * glv.deg], [108.8323824 * glv.deg], [404.684]])
wm = np.array([[0.0]] * 3)
vm = np.array([[0.0]] * 3)
gpsVn = np.array([[0.0]] * 3)
gpsPos = np.array([[0.0]] * 3)
mag = np.array([[0.0]] * 3)

SatNum = 0
res = 0
ts = 0.01
PDOP = 0.0
gpsYaw = 0
wz = 0.0

ravar = CRAvar(5, ts)
ravar.setR0([0.1, 0.01, 1 * glv.deg, 0.1, 3.0])
ravar.setTau([1.0, 1.0, 1.0, 1.0, 1.0])

kf = CTDKF(16, 13, ts)

for i in range(len(datas)):
    wm = [np.array([[datas[i][1]], [datas[i][2]], [datas[i][3]]]) * glv.deg * ts]
    vm = [np.array([[datas[i][4]], [datas[i][5]], [datas[i][6]]]) * ts]
    gpsVn = np.array([[datas[i][10]], [datas[i][11]], [datas[i][12]]])
    PDOP = datas[i][23]
    SatNum = int(datas[i][24])
    gpsPos = np.array([[datas[i][13] * glv.deg], [datas[i][14] * glv.deg], [datas[i][15]]])
    gpsYaw = datas[i][22] * glv.deg
    if gpsYaw > pi:
        gpsYaw = 2 * pi - gpsYaw
    else:
        gpsYaw = -gpsYaw
    mag = np.array([[datas[i][25]], [datas[i][26]], [datas[i][27]]])

    ravar.Update([norm(kf.sins.an), norm(kf.sins.vn), norm(kf.sins.wnb), norm(gpsVn),
                 sqrt(gpsPos[0, 0] ** 2 + gpsPos[1, 0] ** 2) * glv.Re])
    wz = ravar.getindex(2)

    if 1.0 < PDOP < 7.0 and SatNum > 4 and wz < 5 * glv.dps:
        kf.SetMeasGPSVn(gpsVn)
    if 1.0 < PDOP < 7.0 and SatNum > 4 and wz < 10 * glv.dps:
        kf.SetMeasGPSPos(gpsPos)
    if ravar.getindex(0) < 0.15 and wz < 0.15 * glv.dps:
        kf.SetMeasZUPT()
    elif wz < 5 * glv.dps and norm(kf.sins.vn) > 10.0:
        kf.SetMeasMC()
    if gpsYaw != 0 and wz < 1 * glv.dps:
        kf.SetMeasGPSYaw(gpsYaw)

    nm = norm(mag)
    if 200 < nm < 800:
        kf.SetMeasMag(mag)
    kf.Update(wm, vm, ts)
    if kf.yawAlignOK:
        print(kf.sins.pos[0, 0], kf.sins.pos[1, 0], kf.sins.pos[2, 0])
