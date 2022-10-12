from torch import nn
import torch


class Cost(nn.Module):

    def __init__(self, decay_factor, rollout_length, device):
        super(Cost, self).__init__()
        self.decay_factor = decay_factor
        self.rollout_length = rollout_length
        self.device = device

        decay_weight = torch.Tensor(range(self.rollout_length)).to(self.device)
        self.decay_weight = self.decay_factor ** decay_weight

    def forward(self, pred, target):
        """Compute the cost."""

        loss = torch.mean((torch.abs(pred - target)) * self.decay_weight)

        return loss
