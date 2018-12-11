from VDR.MEMS_DATA_DEAL.model import *
from VDR.tools import *
import torch.optim as optim
import torch.utils.data as Data

DIR = r'D:\研究所\重点研究计划\data\MEMS_UBLOX'

Inputs = np.load(DIR + r'\Inputs.npy')
States = np.load(DIR + r'\States.npy')
Truths = np.load(DIR + r'\Truths.npy')

Train_rate = 0.8

INPUT_DIM = 3
OUTPUT_DIM = 3
BATCH_SIZE = 20
EPOCH = 50

Train_Size = int(Inputs.shape[0] * Train_rate)
Test_Size = Inputs.shape[0] - Train_rate

Train_Inputs = torch.from_numpy(Inputs[:Train_Size]).float()
Train_Targets = torch.from_numpy(Truths[:Train_Size]).float()

dataset = Data.TensorDataset(Train_Inputs, Train_Targets)
data_loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE
)

AUA = Angel_Unit(INPUT_DIM, OUTPUT_DIM)
loss_function = nn.MSELoss()
optimizer = optim.Adam(AUA.parameters(), lr=0.01)

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(data_loader):
        input = batch_x[:, :, 4:] * 100

        output = batch_y[:, 1, -3:] - batch_y[:, 0, -3:]
        for i in range(output.shape[0]):
            if output[i, -1] < -180:
                output[i, -1] += 360

        AUA.zero_grad()
        result = AUA(input)

        loss = loss_function(result, output)
        print('Epoch:', epoch, '| Step:', step, '| loss is ', str(loss.item()))
        loss.backward()
        optimizer.step()

torch.save(AUA, DIR + r'AUA.pkl')

