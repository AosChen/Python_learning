import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from scipy.stats import *
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.line1 = nn.Linear(10, 6)
        self.line2 = nn.Linear(7, 3)

    def forward(self, x):
        line1_output = self.line1(x)

        temp = []
        for i in range(20):
            temp.append(kurtosis(x[i]))
        temp = torch.tensor(temp).float()
        line2_input = torch.cat((line1_output, temp.view((temp.shape[0], 1))), 1)

        line2_output = self.line2(line2_input)
        return F.relu(line2_output)


Network = NN()
optimizer = optim.Adam(Network.parameters(), lr=0.01)
loss_function = nn.MSELoss()

BATCH_SIZE = 20
x = torch.linspace(1, 10000, 5000)
x = x.view((5000, 1))
for i in range(9):
    x = torch.cat((x, torch.linspace(1, 10000, 5000).view((5000, 1))), 1)
y = torch.linspace(10000, 1, 5000)
y = y.view((5000, 1))
for i in range(2):
    y = torch.cat((y, torch.linspace(10000, 1, 5000).view((5000, 1))), 1)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        #  Train your datas
        output = Network(batch_x)
        optimizer.zero_grad()
        loss = loss_function(output, batch_y)
        optimizer.step()