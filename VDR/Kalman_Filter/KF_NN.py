import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class KalmanFilter_NN(nn.Module):
    def __init__(self):
        super(KalmanFilter_NN, self).__init__()
        self.line1 = nn.Linear(6, 12)
        self.line2 = nn.Linear(12, 18)
        self.line3 = nn.Linear(18, 6)

    def forward(self, input):
        line1_out = self.line1(input)
        line2_in = F.sigmoid(line1_out)

        line2_out = self.line2(line2_in)
        line3_in = F.tanh(line2_out)

        line3_out = self.line3(line3_in)
        return line3_out
