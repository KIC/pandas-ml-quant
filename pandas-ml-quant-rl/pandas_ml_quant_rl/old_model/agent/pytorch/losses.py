import torch as T
import torch.nn as nn
import torch.functional as F


class LogProbLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, probs, actions, rewards):
        logprob = T.log(probs)
        selected_logprobs = rewards * T.gather(logprob, 1, actions.unsqueeze(1)).squeeze()
        return -selected_logprobs.mean()


class PostSoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.objective = nn.NLLLoss(*args, **kwargs)

    def forward(self, input, target):
        return self.objective(T.log(input), T.log(target))


class QuantileHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantile_tau = None

    def huber_loss(self, x, k=1.0):
        return T.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    def forward(self, input, target):
        assert input.ndim == 3, "expected shape (batch, quantile, action)"
        if self.quantile_tau is None:
            batch, quantile, action = input.shape
            self.quantile_tau = T.FloatTensor([i / quantile for i in range(1, quantile + 1)]).to(input.device)

        td_error = target - input
        huber_loss = self.huber_loss(td_error)
        quantile = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss / 1.0
        loss = quantile.sum(dim=1).mean(dim=1)
        loss = loss.mean()
        return loss



class EntropyLoss(nn.Module):

    def __init__(self, entropy_beta):
        super().__init__()
        self.entropy_beta = entropy_beta

    # baseline is a simple moving average
    def forward(self, logits, baseline):
        log_prob = F.log_softmax(logits, dim=1)
        prob = F.softmax(logits, dim=1)

        entropy = -(prob * log_prob).sum(dim=1).mean()
        entropy_loss = -self.entropy_beta * entropy

        batch_actions_t = 0 # torch.LongTensor(batch_actions).to(device) | batch_actions.append(int(exp.action))
        batch_scale = 0 # FIXME torch.FloatTensor(batch_scales).to(device) | batch_scales.append(exp.reward - baseline)

        log_prob_actions_v = 0 # FIXME batch_scale * log_prob[range(BATCH_SIZE), batch_actions_t]
        loss_policy = -log_prob_actions_v.mean()
        loss = loss_policy + entropy_loss

        return loss


# aliases
HuberLoss = nn.SmoothL1Loss