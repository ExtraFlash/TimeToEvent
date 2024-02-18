import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoNetworks(nn.Module):
    def __init__(self, input_size):
        super(TwoNetworks, self).__init__()
        self.input_size = input_size

        self.t_fc1 = nn.Linear(self.input_size, 1000)
        self.t_bn1 = nn.BatchNorm1d(1000)
        self.t_drop1 = nn.Dropout(0.2)
        self.t_relu1 = nn.ReLU()

        self.t_fc2 = nn.Linear(1000, 1000)
        self.t_bn2 = nn.BatchNorm1d(1000)
        self.t_drop2 = nn.Dropout(0.2)
        self.t_relu2 = nn.ReLU()

        self.t_fc3 = nn.Linear(1000, 1)

        self.p_fc1 = nn.Linear(self.input_size, 1000)
        self.p_bn1 = nn.BatchNorm1d(1000)
        self.p_drop1 = nn.Dropout(0.2)
        self.p_relu1 = nn.ReLU()

        self.p_fc2 = nn.Linear(1000, 1000)
        self.p_bn2 = nn.BatchNorm1d(1000)
        self.p_drop2 = nn.Dropout(0.2)
        self.p_relu2 = nn.ReLU()

        self.p_fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        t = self.t_fc1(x)
        t = self.t_relu1(t)
        t = self.t_fc2(t)
        t = self.t_relu2(t)
        t = self.t_fc3(t)
        t = F.relu(t)

        p = self.p_fc1(x)
        p = self.p_relu1(p)
        p = self.p_fc2(p)
        p = self.p_relu2(p)
        p = self.p_fc3(p)
        p = torch.sigmoid(p)

        return t, p

    def _common_step(self, x):
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        # out = self.drop1(out)

        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)
        # out = self.drop2(out)
        out = self.fc3(out)
        return out
