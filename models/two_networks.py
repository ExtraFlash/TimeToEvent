import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoNetworks(nn.Module):
    def __init__(self, input_size):
        super(TwoNetworks, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.drop1 = nn.Dropout(0.2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.drop2 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        t = self._common_step(x)
        t = F.relu(t)

        p = self._common_step(x)
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
