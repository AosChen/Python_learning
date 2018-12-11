import torch
import torch.nn as nn

# import torch.optim as optim
# import torch.utils.data as Data


'''
Angel_Unit为角度估计模型
输入值：MEMS在一个epoch期间感知到的传感器信号
输出值：三轴姿态角的改变量
目标值：姿态角真值在一个epoch内的改变量，

模型架构：
底层采用CNN来提取传感器的特征，上层采用LSTM累积计算角度的变化
'''


class Angel_Unit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Angel_Unit, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cnn1 = nn.Conv1d(input_dim, input_dim, kernel_size=5, stride=1)
        self.lstm1 = nn.LSTM(input_dim, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 16, batch_first=True)
        self.lstm3 = nn.LSTM(16, output_dim, batch_first=True)

    def forward(self, x):
        temp = x.contiguous().view(x.shape[0], self.input_dim, -1)
        cnn1_output = self.cnn1(temp)
        cnn1_output = cnn1_output.view(cnn1_output.shape[0], -1, cnn1_output.shape[1])

        lstm1_out, hidden1 = self.lstm1(cnn1_output, None)
        lstm2_out, hidden2 = self.lstm2(lstm1_out, None)
        lstm3_out, hidden3 = self.lstm3(lstm2_out, None)

        return lstm3_out[:, -1, :]
