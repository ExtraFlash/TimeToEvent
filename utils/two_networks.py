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
from models.probability_neural_network import ProbabilityNetwork
from models.time_neural_network import TimeNetwork
from losses.my_loss import MyLoss as Loss
from lifelines.utils import concordance_index as ci_lifelines
from losses import ratio


def train_val_test_split(df: pandas.DataFrame):
    df = shuffle(df)
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


def train_model(train_dataset, val_dataset, coef1, coef2):
    y_train = train_dataset.y.numpy()
    # calculate the mean y value of the uncensored samples
    uncensored_mean = np.mean(y_train[y_train[:, 0] == 1][:, 1])
    # calculate the mean y value of the censored samples
    censored_mean = np.mean(y_train[y_train[:, 0] == 0][:, 1])

    n_features = train_dataset.X.shape[1]
    learning_rate = 0.0001
    num_epochs = 3000
    batch_size = 64
    # batch_size = len(train_dataset.X)  # 286

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # define the model
    prob_model = ProbabilityNetwork(n_features)
    time_model = TimeNetwork(n_features)
    criterion = Loss(coef1=coef1, coef2=coef2)
    prob_optimizer = torch.optim.Adam(prob_model.parameters(), lr=learning_rate, weight_decay=0.001)
    time_optimizer = torch.optim.Adam(time_model.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(num_epochs):
        batch_losses_first = []
        batch_losses_second = []
        batch_losses_third = []

        censor_count = 0
        uncensor_count = 0
        epoch_loss = 0.0
        for i, (inputs, outputs) in enumerate(train_loader):
            # forward
            censor_count = torch.sum(Dataset.is_censored(outputs)).item()
            uncensor_count = torch.sum(Dataset.is_uncensored(outputs)).item()
            t_pred = time_model(inputs)
            p_pred = prob_model(inputs)
            # print(t_pred)
            # print(p_pred)
            # print(outputs)
            loss = criterion(t_pred=t_pred, p_pred=p_pred, y_true=outputs,
                             uncensored_mean=uncensored_mean, censored_mean=censored_mean,
                             batch_losses_first=batch_losses_first,
                             batch_losses_second=batch_losses_second,
                             batch_losses_third=batch_losses_third)
            # loss = ratio.RATIO(y=outputs, y_hat=t_pred)
            # # print(f'loss: {loss}')

            # backward
            prob_optimizer.zero_grad()
            time_optimizer.zero_grad()
            loss.backward()
            prob_optimizer.step()
            time_optimizer.step()
            # print(f'p_pred: {p_pred}')
            epoch_loss += outputs.shape[0] * loss.item()
            if math.isinf(loss.item()):
                print('NOWWWWWWWWWWWWWWWWWWWWW')
                print(f'first_loss: {batch_losses_first}')
                print(f'second_loss: {batch_losses_second}')
                print(f'third_loss: {batch_losses_third}')
                return 0
        #     break
        # break
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

        print(
            f'epoch: {epoch}, loss: {(epoch_loss / len(train_dataset))}, censor: {censor_count}, uncensor: {uncensor_count}, mean_p_pred_uncensored: {mean_p_pred_uncensored}, mean_p_pred_censored: {mean_p_pred_censored}, mean_t_pred_uncensored: {mean_t_pred_uncensored}, mean_t_pred_censored: {mean_t_pred_censored}, t_real_mean_uncensored: {t_real_mean_uncensored}, t_real_mean_censored: {t_real_mean_censored}')
        print(f'first_loss: {batch_losses_first}')
        print(f'second_loss: {batch_losses_second}')
        print(f'third_loss: {batch_losses_third}')

    events_obs = val_dataset.y[:, 1].flatten().detach().numpy()
    uncensored_idx = Dataset.is_uncensored(val_dataset.y)
    censored_idx = Dataset.is_censored(val_dataset.y)
    events_obs_uncensored = val_dataset.y[uncensored_idx][:, 1].flatten().detach().numpy()
    events_obs_censored = val_dataset.y[censored_idx][:, 1].flatten().detach().numpy()
    # print(f'events_obs shape: {events_obs.shape}')

    t_pred = time_model(val_dataset.X)
    t_pred_uncensored = t_pred[uncensored_idx].flatten().detach().numpy()
    # print(f't_pred shape: {t_pred.shape}')
    t_pred_censored = t_pred[censored_idx].flatten().detach().numpy()

    p_pred = prob_model(val_dataset.X)
    p_pred_uncensored = p_pred[uncensored_idx].flatten().detach().numpy()
    # print(f'p_pred shape: {p_pred.shape}')
    p_pred_censored = p_pred[censored_idx].flatten().detach().numpy()

    preds = list(t_pred.flatten().detach().numpy())
    # print(f'preds shape: {len(preds)}')
    # get the uncensored and censored samples as a numpy array
    uncensored_idx = Dataset.is_uncensored(val_dataset.y).flatten().detach().numpy()
    # print(f'uncensored_idx shape: {uncensored_idx.shape}')
    events = list(uncensored_idx)
    # print(f'events: {events}')
    print(f'events_obs_uncensored: {events_obs_uncensored}')
    print(f't_pred_uncensored: {t_pred_uncensored}')
    print(f'events: {events}')
    print(f'coef1: {coef1}, coef2: {coef2}, p_pred_last: {p_pred[:3].view(-1)}, loss: {(epoch_loss / len(train_dataset))}, Concordance index: {ci_lifelines(events_obs, preds, events)}')
    print(
        f'epoch: {epoch}, loss: {(epoch_loss / len(train_dataset))}, censor: {censor_count}, uncensor: {uncensor_count},'
        f' p_pred_uncensored_mean: {np.mean(p_pred_uncensored)}, p_pred_censored_mean: {np.mean(p_pred_censored)},'
        f't_pred_uncensored_mean: {np.mean(t_pred_uncensored)}, t_pred_censored_mean: {np.mean(t_pred_censored)},'
        f't_real_uncensored_mean: {np.mean(events_obs_uncensored)}, t_real_censored_mean: {np.mean(events_obs_censored)}')
    print('-------------------------------------------------------------')
    print(f'Concordance index: {ci_lifelines(events_obs, preds, events)}')




    # print(f'y_val_pred: {y_val_pred}')
    # print(f'y_val: {y_val}')
    # print(f'c-index: {ci_lifelines(y_val[:, 1], y_val_pred[:, 0], y_val[:, 0])}')
