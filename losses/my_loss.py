import torch
import torch.nn as nn
from datasets.dataset_real import RealDataset as Dataset
import torch.nn.functional as f
import math


class MyLoss(nn.Module):
    def __init__(self, coef1=1.0, coef2=1.0):
        super(MyLoss, self).__init__()
        self.coef1 = coef1
        self.coef2 = coef2

    def forward(self, t_pred, p_pred, y_true,
                uncensored_mean, censored_mean,
                batch_losses_first=None, batch_losses_second=None, batch_losses_third=None):
        # get the uncensored and censored samples
        uncensored_idx = Dataset.is_uncensored(y_true)
        censored_idx = Dataset.is_censored(y_true)

        loss_censored = 0.0
        # if it remains zero, then loss is a float and will get converted to a tensor
        is_zero = True

        # calculate MSE loss for uncensored samples
        if any(uncensored_idx):
            is_zero = False
            first_loss = f.mse_loss(t_pred[uncensored_idx].view(-1, 1),
                                    Dataset.get_time(y_true)[uncensored_idx].view(-1, 1),
                                    reduction='mean')
            first_normalization = torch.sum(
                torch.square(Dataset.get_time(y_true)[uncensored_idx].view(-1, 1) - uncensored_mean))
            loss_censored += first_loss / first_normalization  # TODO: check this
            # loss_censored += first_loss
            # calculate negative log likelihood for uncensored samples
            if batch_losses_first is not None:
                batch_losses_first.append(first_loss.item() / first_normalization.item())

            second_loss = - torch.mean(torch.log(p_pred[uncensored_idx].view(-1, 1))) * self.coef1
            loss_censored += second_loss
            if math.isinf(second_loss.item()):
                print('second_loss is inf')
                print(f'p_pred: {p_pred[uncensored_idx]}')
            if batch_losses_second is not None:
                batch_losses_second.append(second_loss.item())

        # calculate sum over the censored: (t_pred - t_censored)^2 * sigmoid(t_censored * p_censored - t_pred)
        if any(censored_idx):
            is_zero = False

            first_term = torch.square(t_pred[censored_idx].view(-1, 1) - Dataset.get_time(y_true)[censored_idx].view(-1, 1))
            second_term = f.sigmoid(
                torch.mul(Dataset.get_time(y_true)[censored_idx].flatten(), p_pred[censored_idx].flatten()).view(-1, 1)
                - t_pred[censored_idx].view(-1, 1))
            # loss_censored += f.mse_loss(first_term, second_term, reduction='mean')
            third_loss = torch.mean(torch.mul(first_term, second_term)) * self.coef2
            third_normalization = torch.sum(
                torch.square(Dataset.get_time(y_true)[censored_idx].view(-1, 1) - censored_mean))
            loss_censored += third_loss / third_normalization
            # loss_censored += third_loss
            # print(f'mul dimension: {torch.mul(first_term, second_term).shape}')
            # print(f'mean: {torch.mean(torch.mul(first_term, second_term)).item()}')
            # print(f'mean dimension: {torch.mean(torch.mul(first_term, second_term)).shape}')
            if batch_losses_third is not None:
                batch_losses_third.append(third_loss.item() / third_normalization.item())

        if is_zero:
            loss_censored = torch.tensor(0., torch.float32, requires_grad=True)
            return loss_censored

        return loss_censored
