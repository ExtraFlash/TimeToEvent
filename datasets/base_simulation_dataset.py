import torch
from torch.utils.data import Dataset
import numpy as np
import random


class BaseSimulationDataset(Dataset):

    # CONSTANTS
    C = 2

    def __init__(self,
                 censored_ratio,
                 event_ratio,
                 features_dimension,
                 num_samples,
                 vector_to_multiply=None,
                 intercept=None,
                 should_sample_vector=True):
        """
        :type censored_ratio: float
        :param censored_ratio: Probability for a sample to be censored.
        :param event_ratio: Probability for a sample to have an event.
        :param features_dimension: Amount of features for each sample.
        :param num_samples: Amount of samples.
        :param vector_to_multiply: Vector for linear transformation.
        :param should_sample_vector: Boolean to indicate if vectors should be sampled or given as parameters.
        """
        super(BaseSimulationDataset, self).__init__()
        self.censored_ratio = censored_ratio
        self.event_ratio = event_ratio
        self.features_dimension = features_dimension
        self.num_samples = num_samples
        self.should_sample_vector = should_sample_vector
        if not should_sample_vector:  # if vectors are given as parameters
            self.vector_to_multiply = vector_to_multiply
            self.intercept = intercept
        else:  # if vectors should be sampled
            self.vector_to_multiply = np.random.normal(0, 1, size=self.features_dimension)
            self.intercept = np.random.normal(0, 1)

        # create the data
        self.x = torch.zeros(self.num_samples, self.features_dimension)
        self.y = torch.zeros(self.num_samples, 4)  # [T_c_i, T_e_i, I_c_i, I_e_i]

        for i in range(self.num_samples):
            x_i = np.random.normal(5, 1, size=self.features_dimension)
            t_c_i = np.random.exponential(self.C)
            e_i = np.dot(self.vector_to_multiply, x_i) + self.intercept + np.random.normal(0, 1)
            t_e_i = np.random.normal(e_i, 1)
            # sample a bernoulli variable to decide if an event should be created
            is_event = random.choices([0, 1], weights=[1 - self.event_ratio, self.event_ratio])[0]
            if is_event:
                if t_e_i < t_c_i:  # event no censored
                    self.y[i] = torch.tensor([0, t_e_i, 0, 1])
                    self.x[i] = torch.from_numpy(x_i.astype(np.float32).reshape(1, -1))
                else:  # event censored
                    self.y[i] = torch.tensor([t_c_i, 0, 1, 1])
                    self.x[i] = torch.from_numpy(x_i.astype(np.float32).reshape(1, -1))
            else:  # no event
                self.x[i] = torch.from_numpy(x_i.astype(np.float32).reshape(1, -1))
                self.y[i] = torch.tensor([t_c_i, 0, 1, 0])

    @staticmethod
    def is_sample_event(y):
        if len(y.shape) == 1:
            return y[3] == 1
        return y[:, 3] == 1

    @staticmethod
    def is_sample_censored(y):
        if len(y.shape) == 1:
            return y[2] == 1
        return y[:, 2] == 1

    @staticmethod
    def is_sample_uncensored(y):
        if len(y.shape) == 1:
            return y[2] == 0
        return y[:, 2] == 0

    @staticmethod
    def get_event_time(y):
        if len(y.shape) == 1:
            return y[1]
        return y[:, 1]

    @staticmethod
    def get_censored_time(y):
        if len(y.shape) == 1:
            return y[0]
        return y[:, 0]

    def censored_amount(self):
        """
        :return: Amount of censored samples.
        :return:
        """
        sum_ = 0
        for i in range(self.num_samples):
            if self.is_sample_censored(i):
                sum_ += 1
        return sum_

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]
