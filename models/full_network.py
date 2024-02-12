import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RealNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(RealNeuralNetwork, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()

        self.fc_event_time_hidden = nn.Linear(32, 16)
        self.bn_event_time_hidden = nn.BatchNorm1d(16)
        self.drop_event_time_hidden = nn.Dropout(0.2)
        self.fc_event_time_last = nn.Linear(16, 1)

        self.fc_probability_hidden = nn.Linear(32, 16)
        self.bn_probability_hidden = nn.BatchNorm1d(16)
        self.drop_probability_hidden = nn.Dropout(0.2)
        self.fc_probability_last = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        t = self.fc_event_time_hidden(out)
        t = self.bn_event_time_hidden(t)
        t = F.relu(t)
        t = self.drop_event_time_hidden(t)
        t = self.fc_event_time_last(t)
        t = F.relu(t)

        p = self.fc_probability_hidden(out)
        p = self.bn_probability_hidden(p)
        p = F.relu(p)
        p = self.drop_probability_hidden(p)
        p = self.fc_probability_last(p)
        p = torch.sigmoid(p)

        return t, p

    # def __init__(self, input_size):
    #     super(RealNeuralNetwork, self).__init__()
    #     self.input_size = input_size
    #
    #     self.fc1 = nn.Linear(self.input_size, 1024)
    #     self.bn1 = nn.BatchNorm1d(1024)
    #     self.drop1 = nn.Dropout(0.2)
    #     self.relu1 = nn.ReLU()
    #
    #     self.fc2 = nn.Linear(1024, 512)
    #     self.bn2 = nn.BatchNorm1d(512)
    #     self.drop2 = nn.Dropout(0.2)
    #     self.relu2 = nn.ReLU()
    #
    #     self.fc_event_time_hidden = nn.Linear(512, 32)
    #     self.bn_event_time_hidden = nn.BatchNorm1d(32)
    #     self.drop_event_time_hidden = nn.Dropout(0.2)
    #     self.fc_event_time_last = nn.Linear(32, 1)
    #
    #     self.fc_probability_hidden = nn.Linear(512, 32)
    #     self.bn_probability_hidden = nn.BatchNorm1d(32)
    #     self.drop_probability_hidden = nn.Dropout(0.2)
    #     self.fc_probability_last = nn.Linear(32, 1)
