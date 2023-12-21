from torch.utils.data import Dataset
import numpy as np


class BaseSimulationDataset(Dataset):

    def __init__(self, censored_ratio, event_ratio, features_dimension, num_samples, ):
        super(BaseSimulationDataset, self).__init__()
        self.data = None
        self.labels = None
        self.data_size = None
        self.label_size = None