import torch
from torch.utils.data import Dataset
import numpy as np
import random


class BaseSimulationDataset(Dataset):

    def __init__(self,
                 censored_ratio,
                 event_ratio,
                 features_dimension,
                 num_samples,
                 vector_censored=None,
                 vector_event=None,
                 vector_no_event=None,
                 should_sample_vectors=True):
        """
        :type censored_ratio: float
        :param censored_ratio: Probability for a sample to be censored.
        :param event_ratio: Probability for a sample to have an event.
        :param features_dimension: Amount of features for each sample.
        :param num_samples: Amount of samples.
        :param vector_censored: Vector to be used for censored samples.
        :param vector_event: Vector to be used for samples with an event.
        :param vector_no_event: Vector to be used for samples without an event.
        :param should_sample_vectors: Boolean to indicate if vectors should be sampled or given as parameters.
        """
        super(BaseSimulationDataset, self).__init__()
        self.censored_ratio = censored_ratio
        self.event_ratio = event_ratio
        self.features_dimension = features_dimension
        self.num_samples = num_samples
        self.should_sample_vectors = should_sample_vectors
        if not should_sample_vectors:  # if vectors are given as parameters
            self.vector_censored = vector_censored
            self.vector_event = vector_event
            self.vector_no_event = vector_no_event
        else:  # if vectors should be sampled
            self.sample_vectors()
        # initialize a torch tensor for the data
        self.x = torch.zeros(self.num_samples, self.features_dimension)
        # initialize a torch tensor for the labels: censor_time, event_time, event_indicator, censor_indicator
        self.labels = torch.zeros(self.num_samples, 4)
        # create the data
        self.create_x()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def sample_vectors(self):
        """
        Samples the vectors for the dataset.
        """
        # sample the vectors from a normal distribution
        self.vector_censored = np.random.normal(size=self.features_dimension)
        self.vector_event = np.random.normal(size=self.features_dimension)
        self.vector_no_event = np.random.normal(size=self.features_dimension)

    def create_x(self):
        """
        Creates the data for the dataset.
        """
        # sample the censoring time for each sample
        censor_time = np.random.uniform(size=self.num_samples)
        # sample the event time for each sample
        event_time = np.random.uniform(size=self.num_samples)
        # sample the event indicator for each sample
        event_indicator = np.random.uniform(size=self.num_samples)
        # sample the censor indicator for each sample
        censor_indicator = np.random.uniform(size=self.num_samples)
        # create the labels
        self.labels[:, 0] = censor_time
        self.labels[:, 1] = event_time
        self.labels[:, 2] = event_indicator
        self.labels[:, 3] = censor_indicator
        # create the data
        for i in range(self.num_samples):
            if censor_indicator[i] < self.censored_ratio:
                self.x[i] = torch.tensor(self.vector_censored)
