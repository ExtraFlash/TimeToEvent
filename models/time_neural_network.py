import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeNetwork(nn.Module):
    def __init__(self, input_size):
        super(TimeNetwork, self).__init__()
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

        # self.fc1 = nn.Linear(self.input_size, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.drop1 = nn.Dropout(0.2)
        # self.relu1 = nn.ReLU()
        #
        # self.fc2 = nn.Linear(64, 32)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.drop2 = nn.Dropout(0.2)
        # self.relu2 = nn.ReLU()
        #
        # # add two more layers
        # self.fc3 = nn.Linear(32, 16)
        # self.bn3 = nn.BatchNorm1d(16)
        # self.drop3 = nn.Dropout(0.2)
        # self.relu3 = nn.ReLU()
        #
        # self.fc4 = nn.Linear(16, 1)
        # self.bn4 = nn.BatchNorm1d(1)
        # self.drop4 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        # out = self.drop1(out)

        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)
        # out = self.drop2(out)

        out = self.fc3(out)

        # out = self.fc1(x)
        # # out = self.bn1(out)
        # out = self.relu1(out)
        # # out = self.drop1(out)
        #
        # out = self.fc2(out)
        # # out = self.bn2(out)
        # out = self.relu2(out)
        # # out = self.drop2(out)
        #
        # # add two more layers
        # out = self.fc3(out)
        # # out = self.bn3(out)
        # out = self.relu3(out)
        # # out = self.drop3(out)
        #
        # out = self.fc4(out)
        # # out = self.bn4(out)
        # # out = self.drop4(out)

        t = out
        t = F.relu(t)

        return t
