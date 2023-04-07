import torch
from torch import nn


class Branch(nn.Module):
    """A command-based branch switching network"""

    def __init__(
        self,
        module: nn.Module,
        num_branches: int,
    ):
        """Create different branches of encoders based on the num_branches"""
        super(Branch, self).__init__()

        self.num_branches = num_branches
        self.branches = nn.ModuleList([module for _ in range(self.num_branches)])

    def forward(self, x, command):
        """Forward pass"""

        # x.shape = (batch_size, action_size)
        # command.shape = (batch_size, 1)
        output = []
        for k in range(self.num_branches):
            output.append(self.branches[k](x))

        x = torch.stack(output, dim=1)
        # x.shape = (batch_size, num_branches, action_size)

        # Select the branch based on the command
        x = x[torch.arange(x.shape[0]), command.squeeze(-1)]
        # x.shape = (batch_size, action_size)

        return x
