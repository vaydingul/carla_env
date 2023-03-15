from collections import deque
import time
import torch
import wandb
from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import convert_standard_bev_to_model_bev
from utils.create_video_from_folder import create_video_from_images
from leaderboard.leaderboard_evaluator_local import main as leaderboard_evaluator_main


class Evaluator:
    def __init__(
        self,
        environment,
        leaderboard,
        device,
    ):

        self.environment = environment
        self.leaderboard = leaderboard
        self.device = device

    def evaluate(self, run):

        leaderboard_evaluator_main(
            self.leaderboard, self.device, self.environment
        )

    #     data = self.environment.get_data()
    #     processed_data = self._process_data(data)

    #     for _ in range(self.num_time_step_previous):

    #         self.world_previous_bev_deque.append(processed_data["bev_tensor"])

    #     with torch.no_grad():

    #         while not self.environment.is_done:

    #             t0 = time.time()

    #             # Get data
    #             ego_previous = processed_data["ego_previous"]
    #             bev_tensor = processed_data["bev_tensor"]
    #             navigational_command = processed_data["navigational_command"]
    #             target_location = processed_data["target_location"]
    #             occupancy = processed_data["occupancy"]

    #             # Append world deque
    #             self.world_previous_bev_deque.append(bev_tensor)

    #             # Make it compatible with torch
    #             world_previous_bev = torch.stack(
    #                 list(self.world_previous_bev_deque), dim=1
    #             ).to(self.device)

    #             # Initialize predicted lists
    #             ego_future_location_predicted_list = []
    #             ego_future_yaw_predicted_list = []
    #             ego_future_speed_predicted_list = []
    #             world_future_bev_predicted_list = []
    #             ego_future_action_predicted_list = []

    #             # It is allowed to calculate a new action
    #             if (self.skip_counter == 0) and (self.repeat_counter == 0):

    #                 for k in range(self.num_time_step_future):

    #                     # Calculate encoded world state and predicted future world state
    #                     (
    #                         world_previous_bev_encoded,
    #                         world_future_bev,
    #                     ) = self.world_forward_model(
    #                         world_previous_bev, sample_latent=True
    #                     )

    #                     # Calculate predicted future ego state
    #                     ego_future = self.ego_forward_model(
    #                         ego_previous, navigational_command
    #                     )

    #                     # Calculate predicted future action
    #                     action = self.policy_model(
    #                         ego_previous,
    #                         world_previous_bev_encoded
    #                         if self.use_world_forward_model_encoder_output_as_world_state
    #                         else world_previous_bev,
    #                         navigational_command,
    #                         target_location,
    #                         occupancy,
    #                     )

    #                     # Take sigmoid of world future bev
    #                     world_future_bev = torch.sigmoid(world_future_bev)

    #                     # Append to lists
    #                     ego_future_location_predicted_list.append(
    #                         ego_future["location"]
    #                     )
    #                     ego_future_yaw_predicted_list.append(ego_future["yaw"])
    #                     ego_future_speed_predicted_list.append(ego_future["speed"])
    #                     world_future_bev_predicted_list.append(world_future_bev)
    #                     ego_future_action_predicted_list.append(action)

    #                     # Update ego_previous and world_previous_bev
    #                     ego_previous = ego_future

    #                     world_previous_bev = torch.cat(
    #                         [world_previous_bev[:, 1:], world_future_bev.unsqueeze(1)],
    #                         dim=1,
    #                     )

    #                 # Stack predicted lists
    #                 ego_future_location_predicted = torch.stack(
    #                     ego_future_location_predicted_list, dim=1
    #                 )
    #                 ego_future_yaw_predicted = torch.stack(
    #                     ego_future_yaw_predicted_list, dim=1
    #                 )
    #                 ego_future_speed_predicted = torch.stack(
    #                     ego_future_speed_predicted_list, dim=1
    #                 )
    #                 world_future_bev_predicted = torch.stack(
    #                     world_future_bev_predicted_list, dim=1
    #                 )
    #                 ego_future_action_predicted = torch.stack(
    #                     ego_future_action_predicted_list, dim=1
    #                 )

    #                 # If number of future time steps is greater than 1
    #                 # Calculate policy cost
    #                 if self.num_time_step_future > 1:

    #                     cost = self.cost(
    #                         ego_future_location_predicted,
    #                         ego_future_yaw_predicted,
    #                         ego_future_speed_predicted,
    #                         world_future_bev_predicted,
    #                     )

    #                     cost_dict = {k: v for (k, v) in cost["cost_dict"].items()}

    #                 else:

    #                     cost_dict = {}

    #             # Fetch predicted action
    #             control_selected = ego_future_action_predicted[0][self.skip_counter]

    #             # Convert to environment control
    #             acceleration = control_selected[0].item()
    #             steer = control_selected[1].item()

    #             throttle, brake = acceleration_to_throttle_brake(
    #                 acceleration=acceleration,
    #             )

    #             env_control = [throttle, steer, brake]

    #             # Step environment
    #             self.environment.step(env_control)

    #             # Get data
    #             data = self.environment.get_data()

    #             # Process data
    #             processed_data = self._process_data(data)

    #             # Calculate current FPS
    #             t1 = time.time()
    #             sim_fps = 1 / (t1 - t0)

    #             image_path = self.environment.render(
    #                 simulation_fps=sim_fps,
    #                 frame_counter=self.frame_counter,
    #                 skip_counter=self.skip_counter,
    #                 repeat_counter=self.repeat_counter,
    #                 **cost["cost_dict"],
    #                 cost_viz={  # Some dummy arguments for visualization
    #                     "world_future_bev_predicted": world_future_bev_predicted,
    #                     "mask_dict": cost["mask_dict"],
    #                     "bev_selected_channels": self.bev_selected_channels,
    #                 },  # It looks like there is not any other way
    #                 ego_viz={
    #                     "ego_future_location_predicted": ego_future_location_predicted,
    #                     "control_selected": control_selected,
    #                 },
    #             )
    #             if self.log_video:
    #                 self.log_video_images_path.append(image_path)

    #             # Update counters
    #             self.frame_counter += 1
    #             self.skip_counter = (
    #                 self.skip_counter
    #                 + (self.repeat_counter + 1 == (self.repeat_frames))
    #             ) % self.skip_frames
    #             self.repeat_counter = (self.repeat_counter + 1) % self.repeat_frames

    #             run.log(
    #                 {
    #                     "sim_fps": sim_fps,
    #                     "frame_counter": self.frame_counter,
    #                 }
    #             )
    #     run.log(
    #         {
    #             "SUCCESSFUL": self.environment.is_done
    #             and (not self.environment.is_collided),
    #             "COLLISION": self.environment.is_collided,
    #         }
    #     )
    #     if self.log_video:
    #         create_video_from_images(
    #             images=self.log_video_images_path,
    #             fps=int(
    #                 1
    #                 / (
    #                     self.environment.config["fixed_delta_seconds"]
    #                     * self.repeat_frames
    #                 )
    #             ),
    #             scale=self.log_video_scale,
    #             path=self.environment.renderer_module.save_path,
    #         )
    #         run.log(
    #             {
    #                 "video": wandb.Video(
    #                     f"{self.environment.renderer_module.save_path}/video.mp4"
    #                 )
    #             }
    #         )

    #     self.environment.close()

    # def _process_data(self, data):

    #     processed_data = {}

    #     if "ego" in data.keys():
    #         ego_previous_location = torch.zeros((1, 2), device=self.device)
    #         ego_previous_yaw = torch.zeros((1, 1), device=self.device)
    #         ego_previous_speed = torch.zeros((1, 1), device=self.device)

    #         ego_previous_location[..., 0] = data["ego"]["location_array"][0]
    #         ego_previous_location[..., 1] = data["ego"]["location_array"][1]
    #         ego_previous_yaw[..., 0] = data["ego"]["rotation_array"][-1]
    #         ego_previous_speed[..., 0] = torch.norm(
    #             torch.Tensor(data["ego"]["velocity_array"])
    #         )

    #         ego_previous_location.requires_grad_(True)
    #         ego_previous_yaw.requires_grad_(True)
    #         ego_previous_speed.requires_grad_(True)

    #         ego_previous = {
    #             "location": ego_previous_location,
    #             "yaw": ego_previous_yaw,
    #             "speed": ego_previous_speed,
    #         }

    #         processed_data["ego_previous"] = ego_previous

    #     if "navigation" in data.keys():
    #         target_location = torch.zeros((1, 2), device=self.device)
    #         target_location[..., 0] = data["navigation"][
    #             "waypoint"
    #         ].transform.location.x
    #         target_location[..., 1] = data["navigation"][
    #             "waypoint"
    #         ].transform.location.y
    #         target_location.requires_grad_(True)

    #         navigational_command = torch.zeros((1,), device=self.device)
    #         navigational_command[..., 0] = data["navigation"]["command"].value - 1
    #         navigational_command = torch.nn.functional.one_hot(
    #             navigational_command.long(), num_classes=self.policy_model.command_size
    #         ).float()

    #         processed_data["target_location"] = target_location
    #         processed_data["navigational_command"] = navigational_command

    #     if "bev_world" in data.keys():
    #         bev_tensor = convert_standard_bev_to_model_bev(
    #             data["bev_world"],
    #             agent_channel=self.bev_agent_channel,
    #             vehicle_channel=self.bev_vehicle_channel,
    #             selected_channels=self.bev_selected_channels,
    #             calculate_offroad=self.bev_calculate_offroad,
    #             device=self.device,
    #         )
    #         bev_tensor.requires_grad_(True)

    #         processed_data["bev_tensor"] = bev_tensor

    #     if "occ" in data.keys():
    #         # occupancy = torch.zeros(
    #         #     (1, self.policy_model.occupancy_size), device=self.device
    #         # )
    #         occupancy = (
    #             torch.Tensor(data["occ"]["occupancy"]).unsqueeze(0).to(self.device)
    #         )

    #         if self.binary_occupancy:
    #             occupancy = (occupancy > self.binary_occupancy_threshold).float()

    #         processed_data["occupancy"] = occupancy

    #     return processed_data
