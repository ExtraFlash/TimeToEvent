import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np


# build a neural network model for the simulation dataset
class SimulationNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, output_size):
        super(SimulationNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.first_hidden_size = first_hidden_size
        self.second_hidden_size = second_hidden_size

        self.fc1 = torch.nn.Linear(self.input_size, self.first_hidden_size)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.first_hidden_size, self.second_hidden_size)
        self.relu2 = torch.nn.ReLU()

        self.fc_probability = torch.nn.Linear(self.second_hidden_size, output_size)
        self.fc_event_time = torch.nn.Linear(self.second_hidden_size, output_size)
        # TODO: check if we need a sigmoid here

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        event_probability = self.fc_probability(x)
        event_probability = torch.sigmoid(event_probability)

        event_time = self.fc_event_time(x)
        event_time = F.relu(event_time)

        return event_time, event_probability
