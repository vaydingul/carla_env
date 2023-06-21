from collections import deque
import time
import torch
from torch import nn
import wandb
from utilities.cost_utils import sample_coefficient, transfer_cost_weight
from utilities.kinematic_utils import acceleration_to_throttle_brake
from utilities.model_utils import convert_standard_bev_to_model_bev
from utilities.create_video_from_folder import create_video_from_images
from utilities.train_utils import cat, to, requires_grad, apply_torch_func
from simple_bev.instance import get_translation_and_rotation_from_sensor_list


class Tester:
    def __init__(
        self,
        environment,
        ego_forward_model,
        world_forward_model,
        rgb2bev_model,
        cost,
        cost_weight,
        device,
        optimizer_class,
        optimizer_config,
        batch_size=1,
        action_size=2,
        num_optimization_iteration=30,
        init_action="zeros",
        num_time_step_previous=20,
        num_time_step_future=10,
        skip_frames=1,
        repeat_frames=1,
        log_video=True,
        log_video_scale=0.1,
        gradient_clip=True,
        gradient_clip_type="norm",
        gradient_clip_value=1,
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
        bev_calculate_offroad=False,
    ):
        self.environment = environment
        self.ego_forward_model = ego_forward_model
        self.world_forward_model = world_forward_model
        self.rgb2bev_model = rgb2bev_model
        self.cost = cost
        self.cost_weight = transfer_cost_weight(cost_weight)
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size
        self.action_size = action_size
        self.num_optimization_iteration = num_optimization_iteration
        self.init_action = init_action
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.skip_frames = skip_frames
        self.repeat_frames = repeat_frames
        self.log_video = log_video
        self.log_video_scale = log_video_scale
        self.gradient_clip = gradient_clip
        self.gradient_clip_type = gradient_clip_type
        self.gradient_clip_value = gradient_clip_value
        self.bev_agent_channel = bev_agent_channel
        self.bev_vehicle_channel = bev_vehicle_channel
        self.bev_selected_channels = bev_selected_channels
        self.bev_calculate_offroad = bev_calculate_offroad

        self.ego_forward_model.eval().to(self.device)
        if self.world_forward_model is not None:
            self.world_forward_model.eval().to(self.device)
        self.rgb2bev_model.eval().to(self.device)

        self.frame_counter = 0
        self.skip_counter = 0
        self.repeat_counter = 0

        if self.log_video:
            self.log_video_images_path = []

        self.images_deque = deque(maxlen=self.num_time_step_previous)

        self._reset()

    def test(self, run):
        self.environment.step()
        data = self.environment.get_data()
        processed_data = self._process_data(data)

        for _ in range(self.num_time_step_previous):
            self.images_deque.append(processed_data["images"])

        while not self.environment.is_done:
            t0 = time.time()

            # Get data
            ego_previous = processed_data["ego_previous"]

            images = processed_data["images"]
            translations = processed_data["translations"]
            rotations = processed_data["rotations"]
            intrinsics = processed_data["intrinsics"]

            # bev_tensor = processed_data["bev_tensor"]
            target = processed_data["target"]

            # Append world deque
            self.images_deque.append(images)

            # Make it compatible with torch
            images = (
                torch.stack(list(self.images_deque), dim=1)
                .to(self.device)
                .requires_grad_(True)
            )

            # It is allowed to calculate a new action
            if (self.skip_counter == 0) and (self.repeat_counter == 0):
                out = self._step(
                    ego_previous=ego_previous,
                    images=images,
                    translations=translations,
                    rotations=rotations,
                    intrinsics=intrinsics,
                    target=target,
                )
                ego_future_action_predicted = out["action"]
                world_future_bev_predicted = out["world_future_bev_predicted"]
                mask_dict = out["cost"]["mask_dict"]
                ego_future_location_predicted = out["ego_future_location_predicted"]
                cost = out["cost"]

            # Fetch predicted action
            control_selected = ego_future_action_predicted[0][self.skip_counter]

            # Convert to environment control
            acceleration = control_selected[0].item()
            steer = control_selected[1].item()

            throttle, brake = acceleration_to_throttle_brake(
                acceleration=acceleration,
            )

            if self.frame_counter <= 2:
                throttle = 1.0
            env_control = [throttle, steer, brake]

            # Step environment
            self.environment.step(env_control)

            # Get data
            data = self.environment.get_data()

            # Process data
            processed_data = self._process_data(data)

            # Calculate current FPS
            t1 = time.time()
            sim_fps = 1 / (t1 - t0)

            image_path = self.environment.render(
                simulation_fps=sim_fps,
                frame_counter=self.frame_counter,
                skip_counter=self.skip_counter,
                repeat_counter=self.repeat_counter,
                action=env_control,
                **cost,
                cost_viz={  # Some dummy arguments for visualization
                    "world_future_bev_predicted": world_future_bev_predicted,
                    "mask_dict": mask_dict,
                    "bev_selected_channels": self.bev_selected_channels,
                    "bev_calculate_offroad": self.bev_calculate_offroad,
                },  # It looks like there is not any other way
                ego_viz={
                    "ego_future_location_predicted": ego_future_location_predicted,
                    "control_selected": control_selected,
                },
            )

            if self.log_video:
                self.log_video_images_path.append(image_path)

            self._reset()

            # Update counters
            self.frame_counter += 1
            self.skip_counter = (
                self.skip_counter + (self.repeat_counter + 1 == (self.repeat_frames))
            ) % self.skip_frames
            self.repeat_counter = (self.repeat_counter + 1) % self.repeat_frames

            run.log(
                {
                    "sim_fps": sim_fps,
                    "frame_counter": self.frame_counter,
                }
            )

        run.log(
            {
                "SUCCESSFUL": self.environment.is_done
                and (not self.environment.is_collided),
                "COLLISION": self.environment.is_collided,
            }
        )

        if self.log_video:
            create_video_from_images(
                images=self.log_video_images_path,
                fps=int(
                    1
                    / (
                        self.environment.config["fixed_delta_seconds"]
                        * self.repeat_frames
                    )
                ),
                scale=self.log_video_scale,
                path=self.environment.renderer_module.save_path,
            )
            run.log(
                {
                    "video": wandb.Video(
                        f"{self.environment.renderer_module.save_path}/video.mp4"
                    )
                }
            )

        self.environment.close()

    def _process_data(self, data):
        processed_data = {}

        if "ego" in data.keys():
            ego_previous = {}
            ego_previous["location"] = data["ego"]["location"]
            ego_previous["rotation"] = data["ego"]["rotation"]
            ego_previous["velocity"] = data["ego"]["velocity"]

            ego_previous = apply_torch_func(ego_previous, torch.tensor)
            ego_previous = apply_torch_func(ego_previous, torch.Tensor.view, (1, 1, -1))

            ego_previous = to(ego_previous, self.device)
            requires_grad(ego_previous, True)

            ego_previous["rotation"] = apply_torch_func(
                torch.deg2rad, ego_previous["rotation"]
            )

            processed_data["ego_previous"] = ego_previous

        if "navigation" in data.keys():
            target_location = torch.zeros((1, 1, 2), device=self.device)
            target_yaw = torch.zeros((1, 1, 1), device=self.device)
            target_speed = torch.zeros((1, 1, 1), device=self.device)

            target_location[..., 0] = data["navigation"][
                "waypoint"
            ].transform.location.x
            target_location[..., 1] = data["navigation"][
                "waypoint"
            ].transform.location.y
            target_yaw[..., 0] = (
                data["navigation"]["waypoint"].transform.rotation.yaw * torch.pi / 180
            )

            # target_speed[..., 0] = 5

            target_location.requires_grad_(True)
            target_yaw.requires_grad_(True)
            target_speed.requires_grad_(True)

            target = {
                "location": target_location,
                "yaw": target_yaw,
                "speed": target_speed,
            }
            processed_data["target"] = target

        if ("rgb_front" in data) and ("rgb_right" in data) and ("rgb_left" in data):
            sensor_list = [
                {"id": sensor["id"], **sensor["config"]}
                for sensor in self.environment.sensors
                if ("rgb" in sensor["id"])
            ]

            (
                cameras,
                translations,
                rotations,
                intrinsics,
            ) = get_translation_and_rotation_from_sensor_list(sensor_list)

            images = (
                torch.stack(
                    [
                        torch.from_numpy(data[sensor["id"]]["data"].copy())
                        .float()
                        .permute(2, 0, 1)
                        / 255.0
                        for sensor in sensor_list
                    ],
                    dim=0,
                )
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )

            translations = (
                torch.stack(translations, dim=0)
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            rotations = (
                torch.stack(rotations, dim=0)
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )
            intrinsics = (
                torch.stack(intrinsics, dim=0)
                .unsqueeze(0)
                .to(self.device)
                .requires_grad_(True)
            )

            processed_data["images"] = images
            processed_data["translations"] = translations
            processed_data["rotations"] = rotations
            processed_data["intrinsics"] = intrinsics

        # if "bev_world" in data.keys():
        #     bev_tensor = convert_standard_bev_to_model_bev(
        #         data["bev_world"],
        #         agent_channel=self.bev_agent_channel,
        #         vehicle_channel=self.bev_vehicle_channel,
        #         selected_channels=self.bev_selected_channels,
        #         calculate_offroad=self.bev_calculate_offroad,
        #         device=self.device,
        #     )
        #     bev_tensor.requires_grad_(True)

        #     processed_data["bev_tensor"] = bev_tensor

        return processed_data

    def _forward_ego_forward_model(self, ego_previous):
        ego_state_predicted_list = []

        # location_predicted.append(ego_previous["location"])
        # yaw_predicted.append(ego_previous["yaw"])
        # speed_predicted.append(ego_previous["speed"])

        for i in range(self.num_time_step_future):
            action_ = self.action[:, i : i + 1]  # .clone()

            ego_next = self.ego_forward_model(ego_previous, action_)

            ego_state_predicted_list.append(ego_next)

            ego_previous = ego_next

        ego_state_predicted = cat(ego_state_predicted_list, dim=1)

        return ego_state_predicted

    def _forward_world_forward_model(self, world_previous_bev):
        world_future_bev_predicted_list = []

        # world_future_bev_predicted.append(bev[:, -1].unsqueeze(1))

        for i in range(self.num_time_step_future):
            if self.world_forward_model is not None:
                (_, world_future_bev_predicted) = self.world_forward_model(
                    bev, sample_latent=True
                )
                world_future_bev_predicted = torch.sigmoid(world_future_bev_predicted)
                # world_future_bev_predicted[world_future_bev_predicted > 0.5] = 1
                # world_future_bev_predicted[world_future_bev_predicted <= 0.5] = 0

                bev = torch.cat(
                    [bev[:, 1:], world_future_bev_predicted.unsqueeze(1)], dim=1
                )

                world_future_bev_predicted_list.append(world_future_bev_predicted)

            else:
                world_future_bev_predicted_list.append(world_previous_bev[:, -1])

        world_future_bev_predicted = torch.stack(world_future_bev_predicted_list, dim=1)

        return world_future_bev_predicted

    def _forward_rgb2bev_model(
        self,
        images,
        rotations,
        translations,
        intrinsics,
    ):
        from simple_bev.run_models import run_model_mpc_rgb2bev

        world_previous_bev_predicted_list = []
        for k in range(images.shape[1]):
            world_previous_bev_predicted_list.append(
                run_model_mpc_rgb2bev(
                    model=self.rgb2bev_model,
                    data=(
                        images[:, k],
                        rotations,
                        translations,
                        intrinsics,
                    ),
                    device=self.device,
                ).sigmoid()
            )

        world_previous_bev_predicted = torch.stack(
            world_previous_bev_predicted_list, dim=1
        )

        return world_previous_bev_predicted

    def _forward_cost(
        self,
        ego_future_location_predicted,
        ego_future_yaw_predicted,
        ego_future_speed_predicted,
        world_future_bev_predicted,
        target,
    ):
        target_location = target["location"]
        target_yaw = target["yaw"]
        target_speed = target["speed"]

        # Calculate the cost
        cost = self.cost(
            ego_future_location_predicted,
            ego_future_yaw_predicted,
            ego_future_speed_predicted,
            world_future_bev_predicted,
        )

        # self.predicted_bev = predicted_bev.clone().detach().cpu().numpy()[0]

        loss = torch.tensor(0.0, device=self.device)

        for cost_key in self.cost_weight.keys():
            self.cost_weight[cost_key] = (
                sample_coefficient(
                    self.cost_weight[cost_key]["mean"],
                    self.cost_weight[cost_key]["std"],
                )
                if isinstance(self.cost_weight[cost_key], dict)
                else self.cost_weight[cost_key]
            )

        for cost_key in cost["cost_dict"].keys():
            assert (
                cost_key in self.cost_weight.keys()
            ), f"{cost_key} not in {self.cost_weight.keys()}"

            loss += cost["cost_dict"][cost_key] * self.cost_weight[cost_key]

        ego_location_l1 = torch.nn.functional.l1_loss(
            ego_future_location_predicted,
            target_location.expand(*(ego_future_location_predicted.shape)),
        )

        ego_yaw_l1 = torch.nn.functional.l1_loss(
            torch.cos(ego_future_yaw_predicted),
            torch.cos(target_yaw.expand(*(ego_future_yaw_predicted.shape))),
        )

        ego_yaw_l1 += torch.nn.functional.l1_loss(
            torch.sin(ego_future_yaw_predicted),
            torch.sin(target_yaw.expand(*(ego_future_yaw_predicted.shape))),
        )

        ego_speed_l1 = torch.nn.functional.l1_loss(
            ego_future_speed_predicted,
            target_speed.expand(*(ego_future_speed_predicted.shape)),
        )

        acceleration_jerk = torch.diff(self.action[..., 0], dim=1).square().sum()
        steer_jerk = torch.diff(self.action[..., 1], dim=1).square().sum()

        loss += (
            ego_location_l1 * self.cost_weight["ego_location_l1"]
            + ego_yaw_l1 * self.cost_weight["ego_yaw_l1"]
            + ego_speed_l1 * self.cost_weight["ego_speed_l1"]
            + acceleration_jerk * self.cost_weight["acceleration_jerk"]
            + steer_jerk * self.cost_weight["steer_jerk"]
        )

        return {
            **cost["cost_dict"],
            "mask_dict": cost["mask_dict"],
            "ego_location_l1": ego_location_l1,
            "ego_yaw_l1": ego_yaw_l1,
            "ego_speed_l1": ego_speed_l1,
            "acceleration_jerk": acceleration_jerk,
            "steer_jerk": steer_jerk,
            "loss": loss,
        }

    def _step(self, ego_previous, images, rotations, translations, intrinsics, target):
        world_previous_bev = self._forward_rgb2bev_model(
            images, rotations, translations, intrinsics
        )

        world_future_bev_predicted = self._forward_world_forward_model(
            world_previous_bev
        )

        for _ in range(self.num_optimization_iteration):
            self.optimizer.zero_grad()

            (ego_future_predicted) = self._forward_ego_forward_model(ego_previous)

            ego_future_location_predicted = ego_future_predicted["location_array"][
                ..., :2
            ]
            ego_future_yaw_predicted = ego_future_predicted["rotation_array"][..., 2:3]
            ego_future_speed_predicted = ego_future_predicted["velocity_array"].norm(
                2, -1, True
            )

            cost = self._forward_cost(
                ego_future_location_predicted,
                ego_future_yaw_predicted,
                ego_future_speed_predicted,
                world_future_bev_predicted,
                target,
            )

            loss = cost["loss"]

            loss.backward(retain_graph=True)

            # print(self.action.grad.sum())
            # print(self.action.sum())

            if self.gradient_clip:
                if self.gradient_clip_type == "value":
                    torch.nn.utils.clip_grad_value_(
                        self.action, self.gradient_clip_value
                    )
                elif self.gradient_clip_type == "norm":
                    torch.nn.utils.clip_grad_norm_(
                        self.action, self.gradient_clip_value
                    )
                else:
                    raise ValueError(
                        f"Invalid gradient clip type {self.gradient_clip_type}"
                    )

            self.optimizer.step()

        return {
            "action": self.action,
            "cost": cost,
            "ego_future_location_predicted": ego_future_location_predicted,
            "ego_future_yaw_predicted": ego_future_yaw_predicted,
            "ego_future_speed_predicted": ego_future_speed_predicted,
            "world_future_bev_predicted": world_future_bev_predicted,
        }

    def _reset(self, initial_guess=None):
        if initial_guess is None:
            if self.init_action == "zeros":
                action = torch.zeros(
                    (self.batch_size, self.num_time_step_future, self.action_size),
                    device=self.device,
                    dtype=torch.float32,
                )

            elif self.init_action == "random":
                # Generate random gaussian around 0 with 0.1 std
                action = (
                    torch.randn(
                        (self.batch_size, self.num_time_step_future, self.action_size),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    * 0.2
                )

                # Generate random number between -1 and 1, having the size of (B, T, A)
                # action = (
                #     torch.rand(
                #         (self.batch_size, self.num_time_step_future, self.action_size),
                #         device=self.device,
                #         dtype=torch.float32,
                #     )
                #     * 2
                #     - 1
                # )

                # action = torch.randn(
                #     (self.batch_size, self.num_time_step_future, self.action_size),
                #     device=self.device,
                #     dtype=torch.float32,
                # )

            elif self.init_action == "ones":
                action = torch.ones(
                    (self.batch_size, self.num_time_step_future, self.action_size),
                    device=self.device,
                    dtype=torch.float32,
                )

            else:
                raise NotImplementedError

        else:
            action = torch.tensor(
                initial_guess, device=self.device, dtype=torch.float32
            ).unsqueeze(0)

        self.action = nn.Parameter(action, requires_grad=True)

        self.optimizer = self.optimizer_class((self.action,), **self.optimizer_config)
