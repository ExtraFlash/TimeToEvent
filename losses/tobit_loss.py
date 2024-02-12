import torch
import torch.nn as nn
from datasets.dataset_real import RealDataset as Dataset
import torch.nn.functional as f


class TobitLoss(nn.Module):
    def __init__(self):
        super(TobitLoss, self).__init__()

    def forward(self, t_pred, y_true):
        # calculate tobit regression loss
        uncensored_idx = Dataset.is_uncensored(y_true)
        censored_idx = Dataset.is_censored(y_true)
        loss_censored = 0.0
        loss_uncensored = 0.0

        # calculate MSE loss for uncensored samples
        if any(uncensored_idx):
            loss_uncensored += f.mse_loss(t_pred[uncensored_idx],
                                          Dataset.get_time(y_true)[uncensored_idx].view(-1, 1), reduction='mean')
        if any(censored_idx):
            loss_uncensored += f.mse_loss(t_pred[censored_idx],
                                          Dataset.get_time(y_true)[censored_idx].view(-1, 1), reduction='mean')
            # calculate negative log likelihood for censored samples
            # loss_censored -= torch.mean(torch.exp(t_pred[censored_idx] - Dataset.get_time(y_true)[censored_idx]))
            # TODO: complete this part of the loss
            # loss_censored += torch.mean(torch.log())
        return loss_uncensored + loss_censored