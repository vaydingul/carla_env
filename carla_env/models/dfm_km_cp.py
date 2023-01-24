from torch import nn
import torch
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy


class DecoupledForwardModelKinematicsCoupledPolicy(nn.Module):

    def __init__(self, ego_model, world_model, policy_model):
        super(DecoupledForwardModelKinematicsCoupledPolicy, self).__init__()

        self.ego_model = ego_model
        self.world_model = world_model
        self.policy_model = policy_model

    def forward(
            self,
            ego_state: dict,
            world_state: torch.tensor,
            command,
            target_location,
            occupancy,
            debug=False,
            test=False,
            action_gt=None) -> torch.tensor:

        assert (debug and (action_gt is not None)) or (not debug), "Ground truth action must be provided in debug mode"

        action = self.policy_model(
            ego_state,
            world_state,
            command,
            target_location,
            occupancy) if not debug else action_gt

        if not test:

            (world_future_bev_predicted) = self.world_model(
                world_previous_bev=world_state, sample_latent=True)

            ego_state_next = self.ego_model(ego_state, action)

            return {
                "ego_state_next": ego_state_next,
                "world_state_next": world_future_bev_predicted,
                "action": action}

        else:

            return {"action": action}

    def get_policy_model(self):
        return self.policy_model

    def get_world_model(self):
        return self.world_model

    def get_ego_model(self):
        return self.ego_model


if __name__ == "__main__":
    ego_model = KinematicBicycleModel(dt=1 / 20)
    world_model = WorldBEVModel()
    policy_model = Policy((8, 192, 192), 4, 2)
    inp1 = {"location": torch.randn(10, 2),
            "yaw": torch.randn(10, 1),
            "speed": torch.randn(10, 1)}
    inp2 = torch.randn(10, 8, 192, 192)
    inp3 = torch.randn(10, 6)
    inp4 = torch.randn(10, 2)
    model = DecoupledForwardModelKinematicsCoupledPolicy(
        ego_model, world_model, policy_model)
    out = model(inp1, inp2, inp3, inp4)
    print(out["ego_state_next"])
    print(out["world_state_next"].shape)
    print(out["action"].shape)
