from collections import deque
import torch

from utils.kinematic_utils import acceleration_to_throttle_brake


class Tester:
    def __init__(
        self,
        environment,
        ego_forward_model,
        world_forward_model,
        policy_model,
        cost,
        device,
        num_time_step_previous=20,
        num_time_step_future=10,
        binary_occupancy=False,
        binary_occupancy_threshold=5.0,
        use_world_forward_model_encoder_output_as_world_state=True,
    ):

        self.environment = environment
        self.ego_forward_model = ego_forward_model
        self.world_forward_model = world_forward_model
        self.policy_model = policy_model
        self.cost = cost
        self.device = device
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.binary_occupancy = binary_occupancy
        self.binary_occupancy_threshold = binary_occupancy_threshold
        self.use_world_forward_model_encoder_output_as_world_state = (
            use_world_forward_model_encoder_output_as_world_state
        )

        self.ego_forward_model.eval()
        self.world_forward_model.eval()
        self.policy_model.eval()

        self.frame_counter = 0
        self.skip_counter = 0
        self.repeat_counter = 0

        self.world_previous_bev_deque = deque(maxlen=self.num_time_step_previous)

    def test(self, run):

        self.environment.step()
        data = self.environment.get_data()
        processed_data = self._process_data(data)

        for _ in range(self.num_time_step_previous):

            self.world_previous_bev_deque.append(processed_data["bev_tensor"])

        with torch.no_grad():

            while not self.environment.is_done:

                ego_previous = processed_data["ego_state"]
                bev_tensor = processed_data["bev_tensor"]
                command = processed_data["command"]
                target_location = processed_data["target_location"]
                occupancy = processed_data["occupancy"]

                self.world_previous_bev_deque.append(bev_tensor)

                world_previous_bev = torch.stack(
                    list(self.world_previous_bev_deque), dim=1
                )

                ego_future_location_predicted_list = []
                ego_future_yaw_predicted_list = []
                ego_future_speed_predicted_list = []
                world_future_bev_predicted_list = []
                ego_future_action_predicted_list = []

                for k in range(self.num_time_step_future):

                    (
                        world_previous_bev_encoded,
                        world_future_bev,
                    ) = self.world_forward_model(world_previous_bev, sample_latent=True)

                    ego_future = self.ego_forward_model(ego_previous, command)

                    action = self.policy_model(
                        ego_previous,
                        world_previous_bev_encoded
                        if self.use_world_forward_model_encoder_output_as_world_state
                        else world_previous_bev,
                        command,
                        target_location,
                        occupancy,
                    )

                    world_future_bev = torch.sigmoid(world_future_bev)

                    ego_future_location_predicted_list.append(ego_future["location"])
                    ego_future_yaw_predicted_list.append(ego_future["yaw"])
                    ego_future_speed_predicted_list.append(ego_future["speed"])
                    world_future_bev_predicted_list.append(world_future_bev)
                    ego_future_action_predicted_list.append(action)

                    ego_previous = ego_future

                    world_previous_bev = torch.cat(
                        [world_previous_bev[:, 1:], world_future_bev.unsqueeze(1)],
                        dim=1,
                    )

                ego_future_location_predicted = torch.stack(
                    ego_future_location_predicted_list, dim=1
                )
                ego_future_yaw_predicted = torch.stack(
                    ego_future_yaw_predicted_list, dim=1
                )
                ego_future_speed_predicted = torch.stack(
                    ego_future_speed_predicted_list, dim=1
                )
                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1
                )
                ego_future_action_predicted = torch.stack(
                    ego_future_action_predicted_list, dim=1
                )

                if self.num_time_step_future > 1:

                    cost = self.cost(
                        ego_future_location_predicted,
                        ego_future_yaw_predicted,
                        ego_future_speed_predicted,
                        world_future_bev_predicted,
                    )

                    cost_dict = {k: v for (k, v) in cost["cost_dict"].items()}

                else:

                    cost_dict

                control_selected = ego_future_action_predicted[0][0]

                acceleration = control_selected[0].item()
                steer = control_selected[1].item()

                throttle, brake = acceleration_to_throttle_brake(
                    acceleration=acceleration,
                )

                env_control = [throttle, steer, brake]

                self.environment.step(env_control)

                data = self.environment.get_data()

                processed_data = self._process_data(data)

                self.environment.render()

    def _process_data(self, data):

        pass
        # Convert data dict such that it is compatible with policy models
        # Convert everything to CUDA tensors
