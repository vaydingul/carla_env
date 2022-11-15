from torch import nn
import torch


class DecoupledForwardModelKinematicsPolicy(nn.Module):

    def __init__(self, ego_model, world_model, policy_model, device):
        super(DecoupledForwardModelKinematicsPolicy, self).__init__()

        self.ego_model = ego_model
        self.world_model = world_model
        self.policy_model = policy_model
        self.device = device

    def forward(
            self,
            ego_state: dict,
            world_state: torch.tensor) -> torch.tensor:

        action = self.policy_model(
            torch.cat([v for v in ego_state.values()]), world_state)

        (world_future_bev_predicted, _, _) = self.world_model(
            world_previous_bev=world_state)

        ego_state_next = self.ego_model(ego_state, action)

        return {
            "ego_state_next": ego_state_next,
            "world_state_next": world_future_bev_predicted,
            "action": action}