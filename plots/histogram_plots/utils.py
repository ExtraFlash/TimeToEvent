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
from models.probability_to_time_network import ProbabilityToTimeNetwork

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


def get_model(config, model_name, n_features):
    """
    Get the model class
    :param config: configuration file
    :param model_name: name of the model
    :param n_features: number of features
    :return: model class
    """
    if model_name == 'two_networks':
        return TwoNetworks(n_features)
    elif model_name == 'time_neural_network':
        return TimeNetwork(n_features)
    elif model_name == 'probability_to_time_network':
        return ProbabilityToTimeNetwork(n_features)
    else:
        raise ValueError('Model not found')


def train_model(config_file, identifier, train_dataset, val_dataset, model, criterion,
                dataset_name, model_name, loss_name, coefficients=None):
    """
    Train the model
    :param config_file: path to the configuration file
    :param train_dataset: train dataset
    :param model: model
    :param critertion: loss function
    """
    folder_path = f'{dataset_name}_{model_name}_{loss_name}_{identifier}'
    with open(config_file, encoding='utf-8') as f:
        config = json.load(f)
    # load the configuration
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    optimizer_name = config['optimizer']
    tasks = config['tasks']
    resume_from_checkpoint = config.get('resume_from_checkpoint', False)
    losses_amount = config['losses_characteristics'][loss_name]['losses_amount']

    train_epoch_losses = []  # list of lists, each list contains the corresponding losses for each epoch
    for i in range(losses_amount):
        train_epoch_losses.append([])

    val_epoch_losses = []  # list of lists, each list contains the corresponding losses for each epoch
    for i in range(losses_amount):
        val_epoch_losses.append([])

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not found')
    if resume_from_checkpoint:
        checkpoint = torch.load(f'{folder_path}/checkpoint.pth')
        optimizer.load_state_dict(checkpoint['optim_state'])
        loaded_epoch = checkpoint['epoch']

    y_train = train_dataset.y.numpy()
    is_output_time_probability = False
    # check if the model output is time and probability
    if len(config['models_characteristics'][model_name]['model_output']) == 2:
        is_output_time_probability = True
        if resume_from_checkpoint:
            uncensored_mean = checkpoint['uncensored_mean']
            censored_mean = checkpoint['censored_mean']
        else:
            # calculate the mean y value of the uncensored samples
            uncensored_mean = np.mean(y_train[y_train[:, 0] == 1][:, 1])
            # calculate the mean y value of the censored samples
            censored_mean = np.mean(y_train[y_train[:, 0] == 0][:, 1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train the model
    for epoch in range(num_epochs):
        model.train()  # set the model to training model. because we call model.eval() in the validation loop
        censor_count = 0
        uncensor_count = 0
        # epoch_loss = 0.0
        epoch_cumulative_losses = []  # list where each element is a cumulative not normalized corresponding loss
        for i in range(losses_amount):
            epoch_cumulative_losses.append(0.0)
        epoch_cumulative_amounts = []  # list where each element is a cumulative amount of corresponding samples to normalize the cumulative loss
        for i in range(losses_amount):
            epoch_cumulative_amounts.append(0)

        for i, (inputs, outputs) in enumerate(train_loader):
            censor_count = torch.sum(Dataset.is_censored(outputs)).item()
            uncensor_count = torch.sum(Dataset.is_uncensored(outputs)).item()
            if is_output_time_probability:
                t_pred, p_pred = model(inputs)
                loss = criterion(t_pred=t_pred, p_pred=p_pred, y_true=outputs,
                                 uncensored_mean=uncensored_mean, censored_mean=censored_mean, coefficients=coefficients,
                                 cumulative_losses=epoch_cumulative_losses, cumulative_amounts=epoch_cumulative_amounts)
            else:
                t_pred = model(inputs)
                loss = criterion(t_pred, outputs, coefficients=coefficients,
                                 cumulative_losses=epoch_cumulative_losses, cumulative_amounts=epoch_cumulative_amounts)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # epoch_loss += outputs.shape[0] * loss.item()
            # print(f'epoch_loss: {epoch_loss}')
            # print(f'batch_losses_first: {np.sum(batch_losses_first)}')
            # print(f'batch_losses_second: {np.sum(batch_losses_second)}')
            # print(f'batch_losses_third: {np.sum(batch_losses_third)}')
            # if forth_loss_train is not None:
            #     print(f'batch_losses_forth: {np.sum(batch_losses_forth)}')
        epoch_loss = sum(epoch_cumulative_losses[i] / epoch_cumulative_amounts[i] for i in range(losses_amount))
        for i in range(losses_amount):
            train_epoch_losses[i].append(epoch_cumulative_losses[i] / epoch_cumulative_amounts[i])
        # validation loss
        with torch.no_grad():
            model.eval()
            val_epoch_cumulative_losses = []  # list where each element is a cumulative not normalized corresponding loss
            for i in range(losses_amount):
                val_epoch_cumulative_losses.append(0.0)
            val_epoch_cumulative_amounts = []  # list where each element is a cumulative amount of corresponding samples to normalize the cumulative loss
            for i in range(losses_amount):
                val_epoch_cumulative_amounts.append(0)
            if is_output_time_probability:
                val_t_pred, val_p_pred = model(val_dataset.X)
                criterion(val_t_pred, val_p_pred, val_dataset.y, uncensored_mean, censored_mean, coefficients=coefficients,
                          cumulative_losses=val_epoch_cumulative_losses, cumulative_amounts=val_epoch_cumulative_amounts)
            else:
                val_t_pred = model(val_dataset.X)
                criterion(val_t_pred, val_dataset.y, coefficients=coefficients,
                          cumulative_losses=val_epoch_cumulative_losses, cumulative_amounts=val_epoch_cumulative_amounts)
            # update validation epoch losses
            for i in range(losses_amount):
                val_epoch_losses[i].append(val_epoch_cumulative_losses[i] / val_epoch_cumulative_amounts[i])

        if tasks.get('show_logs', False):
            if is_output_time_probability:
                # get the mean of p_pred over the uncensored samples
                mean_p_pred_uncensored = torch.mean(p_pred[Dataset.is_uncensored(outputs)])
                # get the mean of p_pred over the censored samples
                mean_p_pred_censored = torch.mean(p_pred[Dataset.is_censored(outputs)])
            # get the mean of t_pred over the uncensored samples
            mean_t_pred_uncensored = torch.mean(t_pred[Dataset.is_uncensored(outputs)])
            # get the mean of t_pred over the censored samples
            mean_t_pred_censored = torch.mean(t_pred[Dataset.is_censored(outputs)])
            # get the mean of the real t over the uncensored samples
            t_real_mean_uncensored = torch.mean(outputs[Dataset.is_uncensored(outputs)][:, 1])
            # get the mean of the real t over the censored samples
            t_real_mean_censored = torch.mean(outputs[Dataset.is_censored(outputs)][:, 1])
            print(f'dataset: {dataset_name}, model: {model_name}, loss: {loss_name}')
            if is_output_time_probability:
                print(
                    f'epoch: {epoch}, loss: {epoch_loss}, censor: {censor_count}, uncensor: {uncensor_count}, mean_p_pred_uncensored: {mean_p_pred_uncensored}, mean_p_pred_censored: {mean_p_pred_censored}, mean_t_pred_uncensored: {mean_t_pred_uncensored}, mean_t_pred_censored: {mean_t_pred_censored}, t_real_mean_uncensored: {t_real_mean_uncensored}, t_real_mean_censored: {t_real_mean_censored}')
            else:
                print(
                    f'epoch: {epoch}, loss: {epoch_loss}, censor: {censor_count}, uncensor: {uncensor_count}, mean_t_pred_uncensored: {mean_t_pred_uncensored}, mean_t_pred_censored: {mean_t_pred_censored}, t_real_mean_uncensored: {t_real_mean_uncensored}, t_real_mean_censored: {t_real_mean_censored}')
            for i in range(losses_amount):
                print(f'loss_{i}: {train_epoch_losses[i][-1]}')

    if tasks.get('do_plot_losses', False):
        # make plot for each loss
        for i in range(losses_amount):
            plt.plot(train_epoch_losses[i], label='train')
            plt.plot(val_epoch_losses[i], label='val')
            plt.title(f'Loss {i}')
            plt.legend(loc='upper right')

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.savefig(f'{folder_path}/loss_{i}.png')
            plt.clf()

    if tasks.get('save_model', False):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if is_output_time_probability:
            checkpoint = {
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'uncensored_mean': uncensored_mean,
                'censored_mean': censored_mean
            }
        else:
            checkpoint = {
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()
            }
        # add the epoch to the checkpoint
        if resume_from_checkpoint:
            checkpoint['epoch'] = loaded_epoch + epoch
        else:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, f'{folder_path}/checkpoint.pth')
    if len(config['models_characteristics'][model_name]['model_output']) == 2:
        return model, uncensored_mean, censored_mean
    else:
        return model


def plot_model(config_file, val_dataset, model, criterion,
               dataset_name, model_name, loss_name, identifier='',
               uncensored_mean=None, censored_mean=None):
    """
    Plot the model
    :param config_file: configuration file
    :param val_dataset: validation dataset
    :param model: model
    """
    folder_path = f'{dataset_name}_{model_name}_{loss_name}_{identifier}'
    with open(config_file, encoding='utf-8') as f:
        config = json.load(f)
    tasks = config['tasks']

    model.eval()  # set the model to evaluation mode

    y_true = val_dataset.y

    events_obs = val_dataset.y[:, 1].flatten().detach().numpy()
    uncensored_idx = Dataset.is_uncensored(val_dataset.y)
    censored_idx = Dataset.is_censored(val_dataset.y)
    events_obs_uncensored = val_dataset.y[uncensored_idx][:, 1].flatten().detach().numpy()
    events_obs_censored = val_dataset.y[censored_idx][:, 1].flatten().detach().numpy()

    # check if the model output is time and probability
    if len(config['models_characteristics'][model_name]['model_output']) == 2:
        t_pred, p_pred = model(val_dataset.X)
    else:
        t_pred = model(val_dataset.X)
    t_pred_uncensored = t_pred[uncensored_idx].flatten().detach().numpy()
    # print(f't_pred shape: {t_pred.shape}')
    t_pred_censored = t_pred[censored_idx].flatten().detach().numpy()

    preds = list(t_pred.flatten().detach().numpy())
    # get the uncensored and censored samples as a numpy array
    uncensored_idx = Dataset.is_uncensored(val_dataset.y).flatten().detach().numpy()
    events = list(uncensored_idx)

    ci = ci_lifelines(events_obs, preds, events)
    # check if the model output is time and probability
    if len(config['models_characteristics'][model_name]['model_output']) == 2:
        loss = criterion(t_pred=t_pred, p_pred=p_pred, y_true=y_true,
                         uncensored_mean=uncensored_mean, censored_mean=censored_mean).item()
    else:
        loss = criterion(t_pred, y_true).item()
    # save the concordance index
    if tasks.get('save_results', False):
        with open(f'{folder_path}/results.txt', 'w') as f:
            f.write(f'Concordance index: {ci}\n')
            f.write(f'Loss: {loss}\n')
    # make a histogram plot of (t_pred - t_real) for the uncensored samples and the censored samples
    if tasks.get('do_plot', False):
        plt.hist(t_pred_uncensored - events_obs_uncensored, bins=30, alpha=0.5, label='uncensored')
        plt.hist(t_pred_censored - events_obs_censored, bins=30, alpha=0.5, label='censored')
        plt.legend(loc='upper right')
        plt.title('Histogram of t_pred - t_real')
        plt.savefig(f'{folder_path}/plot.png')


def run_config(config_file, identifier='', coefficients=None):
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
        # count number of censored and uncensored samples in the validation set
        uncensored_count = torch.sum(Dataset.is_uncensored(val_dataset.y)).item()
        censored_count = torch.sum(Dataset.is_censored(val_dataset.y)).item()
        # print(f'uncensored count: {uncensored_count}, censored count: {censored_count}')
        # print(f'val_dataset set size: {val_dataset.X.shape[0]}')

        for loss_name in losses_names:
            criterion = get_loss(loss_name)
            loss_input = config['losses_characteristics'][loss_name]['loss_input']

            for model_name in models_names:
                print(f'dataset: {dataset_name}, model: {model_name}, loss: {loss_name}')
                model_output = config['models_characteristics'][model_name]['model_output']
                # if the loss and model are not compatible
                if loss_input != model_output:
                    continue
                # use dictionary for models, so we don't train the same model twice
                model = get_model(config, model_name, n_features)
                if config.get('resume_from_checkpoint', False):
                    checkpoint = torch.load(f'{dataset_name}_{model_name}_{loss_name}_{identifier}/checkpoint.pth')
                    model.load_state_dict(checkpoint['model_state'])
                    # get the uncensored and censored samples if output is time and probability
                    if len(config['models_characteristics'][model_name]['model_output']) == 2:
                        uncensored_mean = checkpoint['uncensored_mean']
                        censored_mean = checkpoint['censored_mean']

                if tasks.get('do_train', False):
                    if len(config['models_characteristics'][model_name]['model_output']) == 2:
                        model, uncensored_mean, censored_mean = train_model(config_file, identifier, train_dataset, val_dataset,
                                                                            model,
                                                                            criterion,
                                                                            dataset_name, model_name, loss_name,
                                                                            coefficients=coefficients)
                    else:
                        model = train_model(config_file, identifier, train_dataset, val_dataset, model, criterion,
                                            dataset_name, model_name, loss_name)
                # Here we got the trained model, loss and dataset
                # now we can perform the tasks specified in the config
                if tasks.get('do_plot', False):
                    if len(config['models_characteristics'][model_name]['model_output']) == 2:
                        plot_model(config_file, val_dataset, model, criterion,
                                   dataset_name, model_name, loss_name, identifier, uncensored_mean, censored_mean)
                    else:
                        plot_model(config_file, val_dataset, model, criterion,
                                   dataset_name, model_name, loss_name, identifier)


if __name__ == '__main__':
    run_config('config.json')
