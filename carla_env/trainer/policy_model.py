import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from carla_env.sampler.distributed_weighted_sampler import DistributedWeightedSampler

from pathlib import Path
from carla_env.renderer.renderer import Renderer, COLORS
from utils.render_utils import (
    postprocess_bev,
    postprocess_mask,
    postprocess_action,
)

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
        self,
        ego_forward_model,
        world_forward_model,
        policy_model,
        dataloader_train,
        dataloader_val,
        optimizer,
        rank,
        cost,
        cost_weight=None,
        lr_scheduler=None,
        save_interval=5,
        val_interval=5,
        num_time_step_previous=10,
        num_time_step_future=10,
        num_epochs=1000,
        current_epoch=0,
        gradient_clip=True,
        gradient_clip_type="norm",
        gradient_clip_value=0.3,
        binary_occupancy=False,
        binary_occupancy_threshold=5.0,
        use_ground_truth=False,
        use_world_forward_model_encoder_output_as_world_state=False,
        debug_render=False,
        renderer=None,
        save_path=None,
        train_step=0,
        val_step=0,
    ):

        self.ego_forward_model = ego_forward_model
        self.world_forward_model = world_forward_model
        self.policy_model = policy_model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.cost = cost
        self.cost_weight = cost_weight
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.gradient_clip = gradient_clip
        self.gradient_clip_type = gradient_clip_type
        self.gradient_clip_value = gradient_clip_value
        self.binary_occupancy = binary_occupancy
        self.binary_occupancy_threshold = binary_occupancy_threshold
        self.use_ground_truth = use_ground_truth
        self.use_world_forward_model_encoder_output_as_world_state = (
            use_world_forward_model_encoder_output_as_world_state
        )
        self.debug_render = debug_render
        self.renderer = renderer
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step

        # --------------------------------- DDP Setup -------------------------------- #
        self.ego_forward_model.to(self.rank)
        self.world_forward_model.to(self.rank)
        self.policy_model.to(self.rank)
        self.ego_forward_model = DDP(
            self.ego_forward_model, device_ids=[self.rank], find_unused_parameters=False
        )
        self.world_forward_model = DDP(
            self.world_forward_model,
            device_ids=[self.rank],
            find_unused_parameters=False,
        )
        self.policy_model = DDP(
            self.policy_model, device_ids=[self.rank], find_unused_parameters=False
        )

        # ----------------------- Save same values know a prior ---------------------- #

        (self.B_TRAIN, _, self.C, self.H, self.W) = next(iter(dataloader_train))[
            "bev_world"
        ]["bev"].shape

        (self.B_VAL, _, _, _, _) = next(iter(dataloader_val))["bev_world"]["bev"].shape

        assert self.debug_render and (
            self.B_TRAIN == self.B_VAL
        ), "Batch size must be same for train and val in DEBUG mode"
        # ------------------------------ Debug Setup ------------------------------- #

        if self.debug_render:

            self.renderer["width"] = (self.W + 20) * self.num_time_step_future + 200
            self.renderer["height"] = (self.H + 10) * self.B_VAL
            self.renderer = Renderer(config=self.renderer)

        else:

            self.renderer = None

    def train(self, epoch, run):

        logger.info("Training epoch {}".format(epoch))

        self.ego_forward_model.train()
        self.world_forward_model.train()
        self.policy_model.train()

        if isinstance(
            self.dataloader_train.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_train.sampler.set_epoch(epoch)

        for i, (data) in enumerate(self.dataloader_train):

            loss_dict = self.shared_step(i, data)

            loss = loss_dict["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Clip the gradients
            if self.gradient_clip:

                if self.gradient_clip_type == "norm":

                    torch.nn.utils.clip_grad_norm_(
                        self.ego_forward_model.parameters(), self.gradient_clip_value
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.world_forward_model.parameters(), self.gradient_clip_value
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), self.gradient_clip_value
                    )

                else:

                    torch.nn.utils.clip_grad_value_(
                        self.ego_forward_model.parameters(), self.gradient_clip_value
                    )

                    torch.nn.utils.clip_grad_value_(
                        self.world_forward_model.parameters(), self.gradient_clip_value
                    )

                    torch.nn.utils.clip_grad_value_(
                        self.policy_model.parameters(), self.gradient_clip_value
                    )

            self.optimizer.step()

            self.train_step += self.B_TRAIN

            if run is not None:
                run.log(
                    {
                        "train/step": self.train_step,
                        **{f"train/{key}": value for key, value in loss_dict.items()},
                    }
                )

        logger.info("Training epoch {} done".format(epoch))

    def validate(self, epoch, run=None):

        logger.info("Validating epoch {}".format(epoch))

        self.ego_forward_model.eval()
        self.world_forward_model.eval()
        self.policy_model.eval()

        if isinstance(
            self.dataloader_val.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_val.sampler.set_epoch(epoch)

        loss_dict_list = []

        with torch.no_grad():
            for i, (data) in enumerate(self.dataloader_val):

                loss_dict = self.shared_step(i, data)

                loss_dict_list.append(loss_dict)

                self.val_step += self.B_VAL

            loss_dict_mean = {}

            for key in loss_dict_list[0].keys():
                loss_dict_mean[key] = np.mean(
                    [loss_dict[key].item() for loss_dict in loss_dict_list]
                )

            loss_dict = {
                "val/step": self.val_step,
                **{f"val/{key}": value for key, value in loss_dict_mean.items()},
            }

            if run is not None:
                run.log(loss_dict)
                if self.lr_scheduler is not None:
                    run.log({"val/lr": self.lr_scheduler.get_last_lr()[0]})

        logger.info("Validating epoch {} done".format(epoch))

        return loss_dict

    def shared_step(self, i, data):

        # ------------------------------ Data Setup ------------------------------- #
        # World previous BEV
        world_previous_bev = data["bev_world"]["bev"][
            :, : self.num_time_step_previous
        ].to(self.rank)

        # World future BEV
        world_future_bev = data["bev_world"]["bev"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
        ].to(self.rank)
        world_future_bev_predicted_list = []

        world_future_bev_ = world_future_bev.clone()

        # Ego previous location
        ego_previous_location = data["ego"]["location_array"][
            :, self.num_time_step_previous - 1, 0:2
        ].to(self.rank)

        # Ego future location
        ego_future_location = data["ego"]["location_array"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
            0:2,
        ].to(self.rank)
        ego_future_location_predicted_list = []

        # Ego previous yaw
        ego_previous_yaw = torch.deg2rad(
            data["ego"]["rotation_array"][:, self.num_time_step_previous - 1, 2:].to(
                self.rank
            )
        )

        # Ego future yaw
        ego_future_yaw = torch.deg2rad(
            data["ego"]["rotation_array"][
                :,
                self.num_time_step_previous : self.num_time_step_previous
                + self.num_time_step_future,
                2:,
            ].to(self.rank)
        )
        ego_future_yaw_predicted_list = []

        # Ego previous speed
        ego_previous_speed = (
            data["ego"]["velocity_array"][:, self.num_time_step_previous - 1]
            .norm(2, -1, keepdim=True)
            .to(self.rank)
        )

        # Ego future speed
        ego_future_speed = (
            data["ego"]["velocity_array"][
                :,
                self.num_time_step_previous : self.num_time_step_previous
                + self.num_time_step_future,
            ]
            .norm(2, -1, keepdim=True)
            .to(self.rank)
        )
        ego_future_speed_predicted_list = []

        # Ego future action
        ego_future_action = data["ego"]["control_array"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
        ].to(self.rank)

        # Ground-truth action has the form of (throttle, steer, brake)
        # We convert it to (acceleration, brake) for the model
        ego_future_action[..., 0] -= ego_future_action[..., -1]
        ego_future_action = ego_future_action[..., :2]

        ego_future_action_predicted_list = []

        # High-level navigational command
        command = (
            data["navigation_downsampled"]["command"][
                :, self.num_time_step_previous - 1
            ]
            .long()
            .to(self.rank)
        )
        command = F.one_hot(command - 1, num_classes=6).float()

        # Target waypoint location
        target_location = data["navigation_downsampled"]["waypoint"][
            :, self.num_time_step_previous - 1, 0:2
        ].to(self.rank)

        # Occupancy data
        occupancy = data["occ"]["occupancy"][:, self.num_time_step_previous - 1].to(
            self.rank
        )

        if self.binary_occupancy:
            occupancy[occupancy <= self.binary_occupancy_threshold] = 1.0
            occupancy[occupancy > self.binary_occupancy_threshold] = 0.0

        # ------------------------------ Forward Pass ------------------------------ #

        B, S_previous, _, _, _ = world_previous_bev.shape
        _, S_future, _, _, _ = world_future_bev.shape

        # Initialize the ego state
        ego_state_previous = {
            "location": ego_previous_location,
            "yaw": ego_previous_yaw,
            "speed": ego_previous_speed,
        }

        for k in range(self.num_time_step_future):

            if self.use_world_forward_model_encoder_output_as_world_state:

                world_previous_bev_feature = self.world_forward_model(
                    world_previous_bev, sample_latent=True, encoded=True
                )

            else:

                world_previous_bev_feature = world_previous_bev

            world_future_bev_predicted = self.world_forward_model(
                world_previous_bev, sample_latent=True
            )
            world_future_bev_predicted = torch.sigmoid(world_future_bev_predicted)

            action = self.policy_model(
                ego_state=ego_state_previous,
                world_state=world_previous_bev_feature.detach(),
                command=command,
                target_location=target_location,
                occupancy=occupancy,
            )

            ego_state_next = self.ego_forward_model(ego_state_previous, action)

            world_future_bev_predicted_list.append(world_future_bev_predicted)

            ego_future_location_predicted_list.append(ego_state_next["location"])
            ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
            ego_future_speed_predicted_list.append(ego_state_next["speed"])
            ego_future_action_predicted_list.append(action)

            # Update the previous bev
            world_previous_bev = torch.cat(
                (
                    world_previous_bev[:, 1:],
                    world_future_bev[:, k].unsqueeze(1)
                    if self.use_ground_truth
                    else world_future_bev_predicted.unsqueeze(1),
                ),
                dim=1,
            )

            ego_state_previous = ego_state_next

        world_future_bev_predicted = torch.stack(world_future_bev_predicted_list, dim=1)

        ego_future_location_predicted = torch.stack(
            ego_future_location_predicted_list, dim=1
        )

        ego_future_yaw_predicted = torch.stack(ego_future_yaw_predicted_list, dim=1)

        ego_future_speed_predicted = torch.stack(ego_future_speed_predicted_list, dim=1)

        ego_future_action_predicted = torch.stack(
            ego_future_action_predicted_list, dim=1
        )

        if self.num_time_step_future > 1:

            cost = self.cost(
                ego_future_location_predicted,
                ego_future_yaw_predicted,
                ego_future_speed_predicted,
                world_future_bev.requires_grad_(True)
                if self.use_ground_truth
                else world_future_bev_predicted,
            )

            cost_dict = {k: v / (B * S_future) for (k, v) in cost["cost_dict"].items()}

        else:

            cost_dict = {}

        action_mse = F.mse_loss(
            ego_future_action_predicted, ego_future_action, reduction="sum"
        ) / (B * S_future)

        action_l1 = F.l1_loss(
            ego_future_action_predicted, ego_future_action, reduction="sum"
        ) / (B * S_future)

        action_jerk = torch.diff(ego_future_action_predicted, dim=1).square().sum() / (
            B * S_future
        )

        d0 = (
            F.l1_loss(
                target_location, ego_future_location_predicted[:, 0], reduction="none"
            )
            .sum(1, keepdim=True)
            .repeat(1, S_future - 1)
        )  # B x 1

        di = F.l1_loss(
            ego_future_location_predicted[:, 1:],
            ego_future_location_predicted[:, 0:1].repeat(1, S_future - 1, 1),
            reduction="none",
        ).sum(
            2
        )  # B x S

        di_ = F.l1_loss(
            target_location.unsqueeze(1).repeat(1, S_future - 1, 1),
            ego_future_location_predicted[:, 1:],
            reduction="none",
        ).sum(
            2
        )  # B x S

        target_progress = (di / d0).mean()

        target_remainder = (di_ / d0).mean()

        ego_state_mse = torch.tensor(0.0).to(self.rank)

        for key in self.policy_model.module.get_keys():

            if key == "location":
                ego_state_mse += F.l1_loss(
                    ego_future_location_predicted, ego_future_location, reduction="sum"
                ) / (B * S_future)

            elif key == "yaw":
                ego_state_mse += F.l1_loss(
                    torch.cos(ego_future_yaw_predicted),
                    torch.cos(ego_future_yaw),
                    reduction="sum",
                ) / (B * S_future)
                ego_state_mse += F.l1_loss(
                    torch.sin(ego_future_yaw_predicted),
                    torch.sin(ego_future_yaw),
                    reduction="sum",
                ) / (B * S_future)

            elif key == "speed":
                ego_state_mse = F.l1_loss(
                    ego_future_speed_predicted, ego_future_speed, reduction="sum"
                ) / (B * S_future)

            else:

                raise NotImplementedError

        loss = torch.tensor(0.0).to(self.rank)

        for cost_key in cost_dict.keys():

            assert (
                cost_key in self.cost_weight.keys()
            ), f"{cost_key} not in {self.cost_weight.keys()}"

            loss += cost_dict[cost_key] * self.cost_weight[cost_key]

        loss += (
            action_mse * self.cost_weight["action_mse"]
            + action_l1 * self.cost_weight["action_l1"]
            + action_jerk * self.cost_weight["action_jerk"]
            + target_progress * self.cost_weight["target_progress"]
            + target_remainder * self.cost_weight["target_remainder"]
            + ego_state_mse * self.cost_weight["ego_state_mse"]
        )

        loss_dict = {
            **cost_dict,
            "action_mse": action_mse,
            "action_l1": action_l1,
            "action_jerk": action_jerk,
            "target_progress": target_progress,
            "target_remainder": target_remainder,
            "ego_state_mse": ego_state_mse,
            "loss": loss,
        }

        self.render(
            i,
            world_future_bev.requires_grad_(False)
            if self.use_ground_truth
            else world_future_bev_predicted,
            cost,
            ego_future_action_predicted,
            ego_future_action,
        )

        return loss_dict

    def learn(self, run=None):

        # self.epoch = -1
        # loss_dict = self.validate(-1, run)
        # logger.info(f"{'='*10} Initial Validation {'='*10}")
        # logger.info(f"Epoch: Start")
        # for (k, v) in loss_dict.items():
        #     logger.info(f"{k}: {v}")
        # logger.info(f"{'='*10} Initial Validation {'='*10}")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch = epoch
            self.train(epoch, run)

            if (epoch + 1) % self.val_interval == 0:

                loss_dict = self.validate(epoch, run)
                logger.info(f"{'='*10} Validation {'='*10}")
                logger.info(f"Epoch: {epoch}")
                for (k, v) in loss_dict.items():
                    logger.info(f"{k}: {v}")
                logger.info(f"{'='*10} Validation {'='*10}")

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if ((epoch + 1) % self.save_interval == 0) and self.save_path is not None:

                torch.save(
                    {
                        "model_state_dict": self.policy_model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict()
                        if self.lr_scheduler
                        else None,
                        "epoch": epoch,
                        "train_step": self.train_step,
                        "val_step": self.val_step,
                    },
                    self.save_path / Path(f"checkpoint_{epoch}.pt"),
                )

                if run is not None:
                    run.save(str(self.save_path / Path(f"checkpoint_{epoch}.pt")))

    def render(self, i, world_future_bev_predicted, cost, action_pred, action_gt):

        if self.debug_render and self.num_time_step_future > 1:

            for (mask_type, mask) in cost["mask_dict"].items():

                self.renderer.reset()

                for k in range(self.B_TRAIN):

                    cursor_row = self.renderer.get_cursor()

                    for m in range(self.num_time_step_future - 1):

                        bev = postprocess_bev(
                            world_future_bev_predicted[k][m + 1],
                            self.dataloader_train.dataset.bev_selected_channels,
                        )
                        mask_ = postprocess_mask(mask[k][m])
                        action_gt_ = postprocess_action(action_gt[k][m + 1])
                        action_pred_ = postprocess_action(action_pred[k][m + 1])

                        cursor_ = tuple(reversed(self.renderer.get_cursor()))

                        self.renderer.render_overlay_image(
                            bev, mask_, 0.5, 0.5, move_cursor="right"
                        )

                        # Draw ground truth action to the left corner of each bev
                        # as vector

                        self.renderer.render_arrow(
                            start=(cursor_[0] + 50, cursor_[1] + 50),
                            end=(
                                cursor_[0] + 50 + action_gt_[1],
                                cursor_[1] + 50 - action_gt_[0],
                            ),
                            color=COLORS.RED,
                            thickness=1,
                            tip_length=0.5,
                        )

                        self.renderer.render_arrow(
                            start=(cursor_[0] + 50, cursor_[1] + 50),
                            end=(
                                cursor_[0] + 50 + action_pred_[1],
                                cursor_[1] + 50 - action_pred_[0],
                            ),
                            color=COLORS.GREEN,
                            thickness=1,
                            tip_length=0.5,
                        )

                        self.renderer.move_cursor(direction="right", amount=(0, 10))

                    cursor_end = self.renderer.get_cursor()
                    self.renderer.move_cursor(direction="point", amount=cursor_row)
                    self.renderer.move_cursor(direction="down", amount=(self.H + 10, 0))

                cursor_end = (20, cursor_end[1] + 10)
                self.renderer.move_cursor(direction="point", amount=cursor_end)

                for (k, v) in cost["cost_dict"].items():

                    # Format the value such that only 3 decimal places are shown
                    self.renderer.render_text(
                        f"{k}: {v:.3f}",
                        font_scale=1,
                        font_thickness=2,
                        move_cursor="down",
                    )

                self.renderer.render_text(
                    text="Ground Truth",
                    font_color=COLORS.RED,
                    font_scale=1,
                    font_thickness=5,
                )
                self.renderer.render_text(
                    text="Prediction",
                    font_color=COLORS.GREEN,
                    font_scale=1,
                    font_thickness=5,
                )

                # Save the canvas

                info = f"{'training' if self.policy_model.training else 'validation'}-{self.epoch}-{mask_type}-{i}"

                self.renderer.show()
                self.renderer.save(info=info)
