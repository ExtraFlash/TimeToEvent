import torch.nn.functional as f
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


class RATIO(nn.Module):
    def __init__(self):
        super(RATIO, self).__init__()

    def forward(self, t_pred, y_true, cumulative_losses=None, cumulative_amounts=None):
        return ratio_loss(y_true, t_pred, cumulative_losses, cumulative_amounts)


def calculate_tau(y_censored):
    """
    Calculate the censoring rate of the observed competing events
    :param y_censored: time to competing events
    :return: tau
    """
    # convert y_censored from torch tensor to pandas dataframe
    y_censored = pd.DataFrame(y_censored.numpy(), columns=["event", "time"])
    # print(y_censored)
    if len(y_censored.index) == 0:
        tau = float(10 ** 20)  # inf
        return tau
    num, bin_edges = np.histogram(y_censored["time"], bins=30)
    # calculate slope
    y_censored["num"] = num[
        np.clip(np.digitize(y_censored["time"], bin_edges, right=True), a_min=1, a_max=len(num)) - 1]
    # find the slope by regression
    reg = LinearRegression(fit_intercept=False).fit(y_censored["time"].to_frame(), y_censored["num"].to_frame())
    slope = reg.coef_
    # print(f'slope: {slope}')
    if slope == 0:
        tau = 0.0
    else:
        tau = 1 / slope
    return float(tau)


def ratio_loss(y, y_hat,
               cumulative_losses, cumulative_amounts):
    """
    RATIO loss implementation, this loss is implemented with the MSE loss, but it can be changed to any other loss function
    :param y: real TTE
    :param y_hat: predicted TTE
    :return: RATIO loss
    """
    # censored = 0
    # uncensored = 1
    loss = 0.
    uncensored_idx = y[:, 0] == 1
    censored_idx = y[:, 0] == 0
    y_cen, y_hat_cen = y[censored_idx], y_hat[censored_idx]
    # calculate tau
    tau = calculate_tau(y_cen)
    # print(f'tau: {tau}')

    y_hat = y_hat.flatten()
    tau_penalty = 0.0
    if tau > 0:
        tau_penalty = torch.mean(y[uncensored_idx, 1].view(-1, 1)) / tau

    if cumulative_losses is not None:
        current_losses = []  # 0: mse, 1: l1, 2: tau, 3: mse_cen
        for i in range(len(cumulative_losses)):
            current_losses.append(0.0)

    if any(uncensored_idx):
        loss_first = f.mse_loss(y[uncensored_idx, 1].view(-1, 1), y_hat[uncensored_idx].view(-1, 1),
                                reduction='mean') * 1.0
        # loss_second = f.l1_loss(y[uncensored_idx, 1].view(-1, 1), y_hat[uncensored_idx].view(-1, 1),
        #                         reduction='mean') * 0.001
        # loss_third = tau_penalty * 0.001
        # if type(loss_third) == float:
        #     loss_third = torch.tensor(0., torch.float32, requires_grad=True)
        # loss += loss_first + loss_second + loss_third
        loss += loss_first
        if cumulative_losses is not None:
            with torch.no_grad():
                uncensored_amount = sum(uncensored_idx)
                cumulative_losses[0] += uncensored_amount * loss_first.item()
                # cumulative_losses[1] += uncensored_amount * loss_second.item()
                # cumulative_losses[2] += uncensored_amount * loss_third.item()
                cumulative_amounts[0] += uncensored_amount
                # cumulative_amounts[1] += uncensored_amount
                # cumulative_amounts[2] += uncensored_amount
                current_losses[0] = uncensored_amount * loss_first.item()
                # current_losses[1] = uncensored_amount * loss_second.item()
                # current_losses[2] = uncensored_amount * loss_third.item()

        # loss += (f.mse_loss(y[uncensored_idx, 1].view(-1, 1), y_hat[uncensored_idx].view(-1, 1), reduction='mean') + f.l1_loss(
        #     y[uncensored_idx, 1].view(-1, 1),
        #     y_hat[uncensored_idx].view(-1, 1), reduction='mean') + tau_penalty) * 0.001
        # print(f'loss: {loss}')
    if len(y_cen) == 0:
        if type(loss) == float:
            loss = torch.tensor(0., torch.float32, requires_grad=True)
        return loss

    to_change = y_hat_cen.flatten() < y_cen[:, 1].flatten()
    if any(to_change):
        # print(f'loss no normalization: {f.mse_loss(y_cen[to_change, 1].view(-1, 1), y_hat_cen[to_change].view(-1, 1), reduction="sum") * 0.5}')
        # print(f'loss normalization: {f.mse_loss(y_cen[to_change, 1].view(-1, 1), y_hat_cen[to_change].view(-1, 1), reduction="mean") * 0.5}')
        # print(f'normalization amount: {sum(to_change)}')
        loss += f.mse_loss(y_cen[to_change, 1].view(-1, 1), y_hat_cen[to_change].view(-1, 1)) * 0.5
        if cumulative_losses is not None:
            with torch.no_grad():
                cumulative_losses[1] += sum(to_change) * f.mse_loss(y_cen[to_change, 1].view(-1, 1),
                                                                        y_hat_cen[to_change].view(-1, 1)).item() * 0.5
                cumulative_amounts[1] += sum(to_change)
                # print(f'now: {cumulative_amounts[3]}')
                current_losses[1] = sum(to_change) * f.mse_loss(y_cen[to_change, 1].view(-1, 1),
                                                                y_hat_cen[to_change].view(-1, 1)).item() * 0.5

    not_changed = len(y_cen) - sum(to_change)

    if (len(y_hat) - not_changed) != 0:
        loss = (loss * len(y_hat)) / (len(y_hat) - not_changed)
        if cumulative_losses is not None:
            with torch.no_grad():
                cumulative_losses[0] = cumulative_losses[0] - current_losses[0] + current_losses[0] * len(y_hat) / (
                        len(y_hat) - not_changed)
                cumulative_losses[1] = cumulative_losses[1] - current_losses[1] + current_losses[1] * len(y_hat) / (
                        len(y_hat) - not_changed)
                # cumulative_losses[2] = cumulative_losses[2] - current_losses[2] + current_losses[2] * len(y_hat) / (
                #         len(y_hat) - not_changed)
                # cumulative_losses[3] = cumulative_losses[3] - current_losses[3] + current_losses[3] * len(y_hat) / (
                #         len(y_hat) - not_changed)
    else:
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    # print('y_hat:')
    # print(y_hat)
    # print('y:')
    # print(y)
    return loss
