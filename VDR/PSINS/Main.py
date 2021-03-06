import csv
from VDR.PSINS.Algo_Class import *

DIR = r'D:\Project\VisualStudio\PSINS\psins_v1.0'
InputFile = DIR + r'\data1 - 副本.csv'
csv_reader = csv.reader(open(InputFile, 'r'))

kf_writer = csv.writer(open(DIR + r'\kf.csv', 'w', newline=''))
sins_writer = csv.writer(open(DIR + r'\sins.csv', 'w', newline=''))
rt_writer = csv.writer(open(DIR + r'\rt.csv', 'w', newline=''))

timu = 0
ts = 0.01

# 读取数据
index = 0
datas = []
for row in csv_reader:
    if index == 0:
        index += 1
        continue
    else:
        datas.append(list(map(float, row)))

gpspos = np.array([[34.196255 * glv.deg], [108.875677 * glv.deg], [410.70]])
kf = CKalman(15, 6)
kf.Init16(CSINS(A2Q(np.array([[-0.821], [3.690], [6.960]]) * glv.deg), np.array([[0.0], [0.0], [0.0]]), gpspos, timu))

for (data, i) in zip(datas, range(len(datas))):
    timu = data[0]
    wm = np.array([[data[1]], [data[2]], [data[3]]])
    vm = np.array([[data[4]], [data[5]], [data[6]]])
    wm = wm * glv.dps * ts
    vm = vm * ts
    if data[19] > 0 and timu > 60:
        gpsvn = np.array([[data[16]], [data[17]], [data[18]]])
        gpspos = np.array([[data[19]], [data[20]], [data[21]]])
        gpspos[0, 0] *= glv.deg
        gpspos[1, 0] *= glv.deg
        kf.SetMeas(gpsvn, gpspos, timu)

    # kf.TDUpdate([wm], [vm], 1, ts, 10)
    kf.Update([wm], [vm], 1, ts)

    print(kf.sins.pos[0, 0], kf.sins.pos[1, 0], kf.sins.pos[2, 0])
    kf_in = kf.Xk.flatten().tolist()[0]
    for j in range(kf.Pk.shape[0]):
        kf_in.append(kf.Pk[j, j])
    kf_in.append(kf.kftk)
    kf_writer.writerow(kf_in)

    sins_in = kf.sins.att.flatten().tolist()
    sins_in += kf.sins.vn.flatten().tolist()
    sins_in += kf.sins.pos.flatten().tolist()
    sins_in += kf.sins.eb.flatten().tolist()
    sins_in += kf.sins.db.flatten().tolist()
    sins_in.append(kf.sins.tk)
    sins_writer.writerow(sins_in)

    rt_in = kf.Rt.flatten().tolist()
    rt_writer.writerow(rt_in)

    if i % 1000 == 0:
        print(str(i / 100))
