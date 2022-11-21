from torch import nn
import torch
from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy


class DecoupledForwardModelKinematicsPolicy(nn.Module):

    def __init__(self, ego_model, world_model, policy_model):
        super(DecoupledForwardModelKinematicsPolicy, self).__init__()

        self.ego_model = ego_model
        self.world_model = world_model
        self.policy_model = policy_model

    def forward(
            self,
            ego_state: dict,
            world_state: torch.tensor,
            command,
            target_location) -> torch.tensor:

        # Combine second and third dimension of world state
        action = self.policy_model(torch.cat(
            [v for v in ego_state.values()], dim=1), world_state, command, target_location)

        (world_future_bev_predicted) = self.world_model(
            world_previous_bev=world_state, sample_latent=True)

        ego_state_next = self.ego_model(ego_state, action)

        return {
            "ego_state_next": ego_state_next,
            "world_state_next": world_future_bev_predicted,
            "action": action}


if __name__ == "__main__":
    ego_model = KinematicBicycleModelV2(dt=1 / 20)
    world_model = WorldBEVModel()
    policy_model = Policy((8, 192, 192), 4, 2)
    inp1 = {"location": torch.randn(10, 2),
            "yaw": torch.randn(10, 1),
            "speed": torch.randn(10, 1)}
    inp2 = torch.randn(10, 8, 192, 192)
    inp3 = torch.randn(10, 6)
    inp4 = torch.randn(10, 2)
    model = DecoupledForwardModelKinematicsPolicy(
        ego_model, world_model, policy_model)
    out = model(inp1, inp2, inp3, inp4)
    print(out["ego_state_next"])
    print(out["world_state_next"].shape)
    print(out["action"].shape)
