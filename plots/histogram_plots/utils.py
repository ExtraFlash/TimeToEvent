import numpy as np
import math
import pandas
from SurvSet.data import SurvLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from datasets.dataset_real import RealDataset as Dataset

# models
from models.probability_neural_network import ProbabilityNetwork
from models.time_neural_network import TimeNetwork
from models.two_networks import TwoNetworks

# losses
from losses.my_loss import MyLoss
from losses.my_loss_no_sigmoid import MyLossNoSigmoid
from losses.ratio import RATIO
from losses.tobit_loss import TobitLoss

from lifelines.utils import concordance_index as ci_lifelines
from losses import ratio
import json
import os


def train_val_test_split(df: pandas.DataFrame):
    """
    Split the data into train, validation and test sets
    :param df: dataframe
    :return: train, validation and test dataset
    """
    df = shuffle(df, random_state=42)
    df.reset_index(drop=True, inplace=True)
    X = df.drop(['pid', 'event', 'time'], axis=1)
    y = df[['event', 'time']]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # reset index so that stratified shuffle split will work
    df.reset_index(drop=True, inplace=True)

    for dev_index, test_index in split.split(df, df['event']):
        X_dev, X_test = X.iloc[dev_index], X.iloc[test_index]
        y_dev, y_test = y.iloc[dev_index], y.iloc[test_index]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # concat X_dev and y_dev
    df_dev = pandas.concat([X_dev, y_dev], axis=1)

    for train_index, val_index in split.split(df_dev, df_dev['event']):
        X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[val_index]
        y_train, y_val = y_dev.iloc[train_index], y_dev.iloc[val_index]

    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


def get_loss(loss_name):
    """
    Get the loss class
    :param loss_name: name of the loss
    :return: loss class
    """
    if loss_name == 'my_loss':
        return MyLoss()
    elif loss_name == 'my_loss_no_sigmoid':
        return MyLossNoSigmoid()
    elif loss_name == 'ratio':
        return RATIO()
    elif loss_name == 'tobit':
        return TobitLoss()
    else:
        raise ValueError('Loss not found')


def get_model(model_name, n_features):
    """
    Get the model class
    :param model_name: name of the model
    :param n_features: number of features
    :return: model class
    """
    if model_name == 'two_networks':
        return TimeNetwork(n_features)
    elif model_name == 'time_neural_network':
        return TwoNetworks(n_features)
    else:
        raise ValueError('Model not found')


def run_config(config_file):
    """
    Run the configuration file
    :param config_file: path to the configuration file
    """
    with open(config_file, encoding='utf-8') as f:
        config = json.load(f)
    # load datasets, models, losses from config
    tasks = config['tasks']
    datasets_names = config['datasets_names']
    models_names = config['models_names']
    losses_names = config['losses_names']

    # go over all the valid combinations of datasets, models and losses
    for dataset_name in datasets_names:
        loader = SurvLoader()
        df, ref = loader.load_dataset(ds_name=dataset_name).values()
        train_dataset, val_dataset, test_dataset = train_val_test_split(df)
        n_features = train_dataset.X.shape[1]

        for loss_name in losses_names:
            criterion = get_loss(loss_name)
            loss_input = config['losses_input'][loss_name]

            for model_name in models_names:
                model = get_model(model_name, n_features)
                model_output = config['models_output'][model_name]
                # if the loss and model are not compatible
                if loss_input != model_output:
                    continue
                # use dictionary for models, so we don't train the same model twice

