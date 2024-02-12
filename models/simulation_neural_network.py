import torch
import torch.nn.functional as F
import numpy as np


# build a neural network model for the simulation dataset
class SimulationNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, probability_hidden_size, event_time_hidden_size, output_size):
        super(SimulationNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.first_hidden_size = first_hidden_size
        self.second_hidden_size = second_hidden_size
        self.probability_hidden_size = probability_hidden_size
        self.event_time_hidden_size = event_time_hidden_size
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.first_hidden_size)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.first_hidden_size, self.second_hidden_size)
        self.relu2 = torch.nn.ReLU()

        self.fc_probability_hidden = torch.nn.Linear(self.second_hidden_size, probability_hidden_size)
        self.fc_event_time_hidden = torch.nn.Linear(self.second_hidden_size, event_time_hidden_size)

        self.fc_probability_last = torch.nn.Linear(probability_hidden_size, output_size)
        self.fc_event_time_last = torch.nn.Linear(event_time_hidden_size, output_size)
        # TODO: check if we need a sigmoid here

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        p = self.fc_probability_hidden(x)
        p = F.relu(p)
        p = self.fc_probability_last(p)
        p = torch.sigmoid(p)

        t = self.fc_event_time_hidden(x)
        t = F.relu(t)
        t = self.fc_event_time_last(t)
        t = F.relu(t)

        # inf_idx = torch.isinf(-event_time)
        # if any(inf_idx):
        #     print('NOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
        #     print('before relu')
        #     print(event_time)

        return t, p
