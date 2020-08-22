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