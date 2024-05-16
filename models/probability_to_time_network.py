import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilityToTimeNetwork(nn.Module):
    def __init__(self, input_size):
        super(ProbabilityToTimeNetwork, self).__init__()
        self.input_size = input_size

        self.p_fc1 = nn.Linear(self.input_size, 1000)
        self.p_bn = nn.BatchNorm1d(1000)
        self.p_drop = nn.Dropout(0.2)
        self.p_relu1 = nn.ReLU()
        self.p_fc2 = nn.Linear(1000, 1)
        self.p_sigmoid = nn.Sigmoid()

        self.t_fc1 = nn.Linear(self.input_size + 1, 1000)  # input_size + 1 because we concatenate the probability
        self.t_relu1 = nn.ReLU()
        self.t_fc2 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        p = self.p_fc1(x)
        p = self.p_bn(p)
        p = self.p_drop(p)
        p = self.p_relu1(p)
        p_no_sigmoid = self.p_fc2(p)
        p_sigmoid = self.p_sigmoid(p_no_sigmoid)

        t = self.t_fc1(torch.cat((x, p_no_sigmoid), 1))
        t = self.t_relu1(t)
        t = self.t_fc2(t)
        t = self.relu(t)

        return t, p_sigmoid
