import torch
import torch.nn as nn
from datasets.dataset_real import RealDataset as Dataset
import torch.nn.functional as f
import math
from losses import ratio


class MyLossNoSigmoid(nn.Module):
    def __init__(self, coef1=1.0, coef2=0.5):
        super(MyLossNoSigmoid, self).__init__()
        self.coef1 = coef1
        self.coef2 = coef2

    def forward(self, t_pred, p_pred, y_true,
                uncensored_mean, censored_mean,
                coefficients=None,
                cumulative_losses=None, cumulative_amounts=None):
        # get the uncensored and censored samples
        if coefficients is None:
            coefficients = [1.0, 1.0, 1.0]
        uncensored_idx = Dataset.is_uncensored(y_true)
        censored_idx = Dataset.is_censored(y_true)

        loss_censored = 0.0
        # if it remains zero, then loss is a float and will get converted to a tensor
        is_zero = True

        tau = ratio.calculate_tau(y_true[censored_idx])
        # print(f'tau: {tau}')

        tau_penalty = 0.0
        if tau > 0:
            tau_penalty = torch.mean(y_true[uncensored_idx, 1].view(-1, 1)) / tau

        # calculate MSE loss for uncensored samples
        if any(uncensored_idx):
            is_zero = False
            first_loss = f.mse_loss(t_pred[uncensored_idx].view(-1, 1),
                                     Dataset.get_time(y_true)[uncensored_idx].view(-1, 1), reduction='mean') * coefficients[0]
            first_normalization = torch.sum(
                torch.square(Dataset.get_time(y_true)[uncensored_idx].view(-1, 1) - uncensored_mean))
            loss_censored += first_loss / first_normalization  # TODO: check this
            # loss_censored += first_loss
            # calculate negative log likelihood for uncensored samples
            if cumulative_losses is not None:
                with torch.no_grad():
                    cumulative_losses[0] += sum(uncensored_idx) * first_loss.item() / first_normalization.item()
                    cumulative_amounts[0] += sum(uncensored_idx)

            second_loss = - torch.mean(torch.log(p_pred[uncensored_idx].view(-1, 1))) * coefficients[1]
            loss_censored += second_loss
            if math.isinf(second_loss.item()):
                print('second_loss is inf')
                print(f'p_pred: {p_pred[uncensored_idx]}')
            if cumulative_losses is not None:
                with torch.no_grad():
                    cumulative_losses[1] += sum(uncensored_idx) * second_loss.item()
                    cumulative_amounts[1] += sum(uncensored_idx)

        # calculate sum over the censored where we estimated lower than the censoring time: (t_pred - t_censored)^2 * p_pred
        # get the indices of the censored samples where the predicted time is lower than the censoring time
        to_change_idx = t_pred.flatten() < Dataset.get_time(y_true).flatten()
        to_change_idx = to_change_idx & censored_idx
        if any(to_change_idx):
            is_zero = False

            first_term = torch.square(
                t_pred[to_change_idx].view(-1, 1) - Dataset.get_time(y_true)[to_change_idx].view(-1, 1))
            second_term = p_pred[to_change_idx].view(-1, 1)
            # loss_censored += f.mse_loss(first_term, second_term, reduction='mean')
            third_loss = torch.mean(torch.mul(first_term, second_term)) * coefficients[2]
            third_normalization = torch.sum(
                torch.square(Dataset.get_time(y_true)[censored_idx].view(-1, 1) - censored_mean))
            loss_censored += third_loss / third_normalization
            # loss_censored += third_loss
            # print(f'mul dimension: {torch.mul(first_term, second_term).shape}')
            # print(f'mean: {torch.mean(torch.mul(first_term, second_term)).item()}')
            # print(f'mean dimension: {torch.mean(torch.mul(first_term, second_term)).shape}')
            if cumulative_losses is not None:
                with torch.no_grad():
                    cumulative_losses[2] += sum(to_change_idx) * third_loss.item() / third_normalization.item()
                    cumulative_amounts[2] += sum(to_change_idx)

        if is_zero:
            loss_censored = torch.tensor(0., torch.float32, requires_grad=True)
            return loss_censored

        return loss_censored
