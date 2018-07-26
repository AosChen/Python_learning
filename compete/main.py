import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn import preprocessing


class MyNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, output_dim)

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer1_out = F.sigmoid(layer1_out)
        layer2_out = self.layer2(layer1_out)
        layer2_out = F.tanh(layer2_out)
        result = F.sigmoid(self.out(layer2_out)).view(-1, 8)
        return result

MY_NN = MyNN(10, 8)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(MY_NN.parameters(), lr=0.001)

b_x = Variable(torch.from_numpy(np.load(r'D:\研究所\\比赛\train\Data_X.npy')), requires_grad=True)
b_y = Variable(torch.from_numpy(np.load(r'D:\研究所\\比赛\train\Data_Y.npy')))
b_y.reshape(16310, 6000, 1)

# torch_dataset = Data.TensorDataset(Data_X, Data_Y)
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=1,
#     shuffle=True,
#     # num_workers=2
# )
#
enc = preprocessing.OneHotEncoder()
nums_class = 8
model2train = [[i + 1] for i in range(nums_class)]
enc.fit(model2train)
mode_onehot = [enc.transform([[1]]).toarray(), enc.transform([[2]]).toarray(), enc.transform([[3]]).toarray(),
               enc.transform([[4]]).toarray(), enc.transform([[5]]).toarray(), enc.transform([[6]]).toarray(),
               enc.transform([[7]]).toarray(), enc.transform([[8]]).toarray()]
mode_onehot = np.array(mode_onehot)


for i in range(16310):
    for j in range(6000):
        input_x = b_x[i][j]
        # input_y = torch.from_numpy(mode_onehot[int(b_y[i][j]) - 1][0]).view(-1, 8).long()
        input_y = b_y[i][j].view(1).long()
        out = MY_NN(input_x)
        print("out", out)
        print("input_y",input_y)
        loss = loss_function(out, input_y)
        print(i, j, loss)
        print('.....................................')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(MY_NN.state_dict(), 'net_params.pkl')