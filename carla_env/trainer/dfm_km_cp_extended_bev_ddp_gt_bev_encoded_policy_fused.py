from carla_env.bev import BirdViewProducer
import logging
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
        self,
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        gpu_id,
        cost,
        cost_weight=None,
        num_time_step_previous=10,
        num_time_step_future=10,
        binary_occupancy=True,
        num_epochs=1000,
        current_epoch=0,
        lr_scheduler=None,
        gradient_clip_type="norm",
        gradient_clip_value=0.3,
        save_path=None,
        train_step=0,
        val_step=0,
        bev_width=192,
        bev_height=192,
        debug_render=False,
        save_interval=5,
    ):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.cost = cost
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.binary_occupancy = binary_occupancy
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.lr_scheduler = lr_scheduler
        self.gradient_clip_type = gradient_clip_type
        self.gradient_clip_value = gradient_clip_value
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.debug_render = debug_render
        self.save_interval = save_interval

        self.cost_weight = {}
        self.cost_weight["road_cost_weight"] = 0.005
        self.cost_weight["road_on_cost_weight"] = 0.005
        self.cost_weight["road_off_cost_weight"] = 0.005
        self.cost_weight["road_green_cost_weight"] = 0.005
        self.cost_weight["road_red_yellow_cost_weight"] = 0.005
        self.cost_weight["lane_cost_weight"] = 0.005
        self.cost_weight["vehicle_cost_weight"] = 0.005
        self.cost_weight["offroad_cost_weight"] = 0.005
        self.cost_weight["action_mse_weight"] = 0.005
        self.cost_weight["action_jerk_weight"] = 0.005
        self.cost_weight["target_progress_weight"] = 0.005
        self.cost_weight["target_remainder_weight"] = 0.005
        self.cost_weight["ego_state_mse_weight"] = 0.005
        self.cost_weight["world_state_mse_weight"] = 0.005

        if cost_weight is not None:
            for (k, v) in cost_weight.items():
                self.cost_weight[k] = v

        self.model.to(self.gpu_id)
        self.model = DDP(
            self.model, device_ids=[self.gpu_id], find_unused_parameters=True
        )

        debug_path = Path("figures/env_debug/train_policy")

        date_ = Path(datetime.today().strftime("%Y-%m-%d"))
        time_ = Path(datetime.today().strftime("%H-%M-%S"))

        self.debug_path = debug_path / date_ / time_
        self.debug_path.mkdir(parents=True, exist_ok=True)

    def train(self, epoch, run):

        self.model.train()
        self.dataloader_train.sampler.set_epoch(epoch)

        for i, (data) in enumerate(self.dataloader_train):

            loss_dict = self.shared_step(i, data, mode="train")

            loss = loss_dict["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Clip the gradients
            if self.gradient_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_value
                )
            else:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip_value
                )

            self.optimizer.step()

            if run is not None:
                run.log(
                    {
                        "train/step": self.train_step,
                        "train/road_cost": loss_dict["road_cost"],
                        "train/road_on_cost": loss_dict["road_on_cost"],
                        "train/road_off_cost": loss_dict["road_off_cost"],
                        "train/road_red_yellow_cost": loss_dict["road_red_yellow_cost"],
                        "train/road_green_cost": loss_dict["road_green_cost"],
                        "train/lane_cost": loss_dict["lane_cost"],
                        "train/vehicle_cost": loss_dict["vehicle_cost"],
                        "train/offroad_cost": loss_dict["offroad_cost"],
                        "train/action_mse": loss_dict["action_mse"],
                        "train/action_jerk": loss_dict["action_jerk"],
                        "train/target_progress": loss_dict["target_progress"],
                        "train/target_remainder": loss_dict["target_remainder"],
                        "train/ego_state_mse": loss_dict["ego_state_mse"],
                        "train/loss": loss,
                    }
                )

    def validate(self, run=None):

        self.model.eval()

        road_cost_list = []
        road_on_cost_list = []
        road_off_cost_list = []
        road_red_yellow_cost_list = []
        road_green_cost_list = []
        lane_cost_list = []
        vehicle_cost_list = []
        offroad_cost_list = []
        action_mse_list = []
        action_jerk_list = []
        target_progress_list = []
        target_remainder_list = []
        ego_state_mse_list = []
        loss_list = []

        with torch.no_grad():
            for i, (data) in enumerate(self.dataloader_val):

                loss_dict = self.shared_step(i, data, mode="val")

                road_cost_list.append(loss_dict["road_cost"].item())
                road_on_cost_list.append(loss_dict["road_on_cost"].item())
                road_off_cost_list.append(loss_dict["road_off_cost"].item())
                road_red_yellow_cost_list.append(
                    loss_dict["road_red_yellow_cost"].item()
                )
                road_green_cost_list.append(loss_dict["road_green_cost"].item())
                lane_cost_list.append(loss_dict["lane_cost"].item())
                vehicle_cost_list.append(loss_dict["vehicle_cost"].item())
                offroad_cost_list.append(loss_dict["offroad_cost"].item())
                action_mse_list.append(loss_dict["action_mse"].item())
                action_jerk_list.append(loss_dict["action_jerk"].item())
                target_progress_list.append(loss_dict["target_progress"].item())
                target_remainder_list.append(loss_dict["target_remainder"].item())
                ego_state_mse_list.append(loss_dict["ego_state_mse"].item())
                loss_list.append(loss_dict["loss"].item())

            road_cost_mean = np.mean(road_cost_list)
            road_on_cost_mean = np.mean(road_on_cost_list)
            road_off_cost_mean = np.mean(road_off_cost_list)
            road_red_yellow_cost_mean = np.mean(road_red_yellow_cost_list)
            road_green_cost_mean = np.mean(road_green_cost_list)
            lane_cost_mean = np.mean(lane_cost_list)
            vehicle_cost_mean = np.mean(vehicle_cost_list)
            offroad_cost_mean = np.mean(offroad_cost_list)
            action_mse_mean = np.mean(action_mse_list)
            action_jerk_mean = np.mean(action_jerk_list)
            target_progress_mean = np.mean(target_progress_list)
            target_remainder_mean = np.mean(target_remainder_list)
            ego_state_mse_mean = np.mean(ego_state_mse_list)
            loss_mean = np.mean(loss_list)

            loss_dict = {
                "val/step": self.val_step,
                "val/road_cost": road_cost_mean,
                "val/road_on_cost": road_on_cost_mean,
                "val/road_off_cost": road_off_cost_mean,
                "val/road_red_yellow_cost": road_red_yellow_cost_mean,
                "val/road_green_cost": road_green_cost_mean,
                "val/lane_cost": lane_cost_mean,
                "val/vehicle_cost": vehicle_cost_mean,
                "val/offroad_cost": offroad_cost_mean,
                "val/action_mse": action_mse_mean,
                "val/action_jerk": action_jerk_mean,
                "val/target_progress": target_progress_mean,
                "val/target_remainder": target_remainder_mean,
                "val/ego_state_mse": ego_state_mse_mean,
                "val/loss": loss_mean,
            }

            if run is not None:
                run.log(loss_dict)
                if self.lr_scheduler is not None:
                    run.log({"val/lr": self.lr_scheduler.get_last_lr()[0]})

            return loss_dict

    def shared_step(self, i, data, mode="train"):

        world_previous_bev = data["bev_world"]["bev"][
            :, : self.num_time_step_previous
        ].to(self.gpu_id)
        world_future_bev = data["bev_world"]["bev"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
        ].to(self.gpu_id)

        (_, S_previous, _, _, _) = world_previous_bev.shape
        (B, S_future, C, H, W) = world_future_bev.shape
        world_future_bev_ = world_future_bev.clone()
        world_future_bev_predicted_list = []

        ego_previous_location = data["ego"]["location_array"][
            :, self.num_time_step_previous - 1, 0:2
        ].to(self.gpu_id)
        ego_future_location = data["ego"]["location_array"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
            0:2,
        ].to(self.gpu_id)
        ego_future_location_predicted_list = []

        ego_previous_yaw = torch.deg2rad(
            data["ego"]["rotation_array"][:, self.num_time_step_previous - 1, 2:].to(
                self.gpu_id
            )
        )
        ego_future_yaw = torch.deg2rad(
            data["ego"]["rotation_array"][
                :,
                self.num_time_step_previous : self.num_time_step_previous
                + self.num_time_step_future,
                2:,
            ].to(self.gpu_id)
        )
        ego_future_yaw_predicted_list = []

        ego_previous_speed = (
            data["ego"]["velocity_array"][:, self.num_time_step_previous - 1]
            .norm(2, -1, keepdim=True)
            .to(self.gpu_id)
        )
        ego_future_speed = (
            data["ego"]["velocity_array"][
                :,
                self.num_time_step_previous : self.num_time_step_previous
                + self.num_time_step_future,
            ]
            .norm(2, -1, keepdim=True)
            .to(self.gpu_id)
        )
        ego_future_speed_predicted_list = []

        ego_future_action = data["ego"]["control_array"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
        ].to(self.gpu_id)
        ego_future_action_predicted_list = []

        command = (
            data["navigation_downsampled"]["command"][
                :, self.num_time_step_previous - 1
            ]
            .long()
            .to(self.gpu_id)
        )
        command = F.one_hot(command - 1, num_classes=6).float()
        target_location = data["navigation_downsampled"]["waypoint"][
            :, self.num_time_step_previous - 1, 0:2
        ].to(self.gpu_id)

        occupancy = data["occ"]["occupancy"][:, self.num_time_step_previous - 1].to(
            self.gpu_id
        )

        if self.binary_occupancy:
            occupancy[occupancy <= 5] = 1.0
            occupancy[occupancy > 5] = 0.0

        ego_state_previous = {
            "location": ego_previous_location,
            "yaw": ego_previous_yaw,
            "speed": ego_previous_speed,
        }

        for k in range(self.num_time_step_future):

            # Predict the future bev
            # output = self.model(
            #     ego_state_previous,
            #     world_previous_bev,
            #     command,
            #     target_location,
            #     occupancy,
            # )

            action = self.model.module.get_policy_model()(
                ego_state_previous,
                self.model.module.get_world_model().world_previous_bev_encoder(
                    world_previous_bev.view(B, -1, H, W)
                ),
                command,
                target_location,
                occupancy,
            )
            # world_state_next = output["world_state_next"]
            ego_state_next = self.model.module.get_ego_model()(
                ego_state_previous, action
            )

            # world_future_bev_predicted = torch.sigmoid(
            #     world_state_next)

            # world_future_bev_predicted_list.append(
            #     world_future_bev_predicted)

            ego_future_location_predicted_list.append(ego_state_next["location"])
            ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
            ego_future_speed_predicted_list.append(ego_state_next["speed"])
            ego_future_action_predicted_list.append(action)

            # Update the previous bev
            world_previous_bev = torch.cat(
                (
                    world_previous_bev[:, 1:],
                    world_future_bev[:, k].unsqueeze(1).clone(),
                ),
                dim=1,
            )

            ego_state_previous = ego_state_next

        # world_future_bev_predicted = torch.stack(
        #     world_future_bev_predicted_list, dim=1)

        ego_future_location_predicted = torch.stack(
            ego_future_location_predicted_list, dim=1
        )

        ego_future_yaw_predicted = torch.stack(ego_future_yaw_predicted_list, dim=1)

        ego_future_speed_predicted = torch.stack(ego_future_speed_predicted_list, dim=1)

        ego_future_action_predicted = torch.stack(
            ego_future_action_predicted_list, dim=1
        )

        if self.num_time_step_future > 1:
            world_future_bev.requires_grad = True
            cost = self.cost(
                ego_future_location_predicted,
                ego_future_yaw_predicted,
                ego_future_speed_predicted,
                world_future_bev,
            )

            road_cost = cost["cost"][0] / (B * S_future)
            road_on_cost = cost["cost"][1] / (B * S_future)
            road_off_cost = cost["cost"][2] / (B * S_future)
            road_red_yellow_cost = cost["cost"][3] / (B * S_future)
            road_green_cost = cost["cost"][4] / (B * S_future)
            lane_cost = cost["cost"][5] / (B * S_future)
            vehicle_cost = cost["cost"][6] / (B * S_future)
            offroad_cost = cost["cost"][7] / (B * S_future)
        else:

            cost = {}
            road_cost = torch.tensor(0.0).to(self.gpu_id)
            road_on_cost = torch.tensor(0.0).to(self.gpu_id)
            road_off_cost = torch.tensor(0.0).to(self.gpu_id)
            road_red_yellow_cost = torch.tensor(0.0).to(self.gpu_id)
            road_green_cost = torch.tensor(0.0).to(self.gpu_id)
            lane_cost = torch.tensor(0.0).to(self.gpu_id)
            vehicle_cost = torch.tensor(0.0).to(self.gpu_id)
            offroad_cost = torch.tensor(0.0).to(self.gpu_id)

        ego_future_action[..., 0] -= ego_future_action[..., -1]

        action_mse = F.mse_loss(
            ego_future_action_predicted, ego_future_action[..., :2], reduction="sum"
        ) / (B * S_future)

        action_jerk = torch.diff(ego_future_action_predicted, dim=1).square().sum() / (
            B * S_future
        )

        d0 = (
            F.l1_loss(
                target_location, ego_future_location_predicted[:, 0], reduction="none"
            )
            .sum(1, keepdim=True)
            .repeat(1, self.num_time_step_future - 1)
        )  # B x 1

        di = F.l1_loss(
            ego_future_location_predicted[:, 1:],
            ego_future_location_predicted[:, 0:1].repeat(
                1, self.num_time_step_future - 1, 1
            ),
            reduction="none",
        ).sum(
            2
        )  # B x S

        di_ = F.l1_loss(
            target_location.unsqueeze(1).repeat(1, self.num_time_step_future - 1, 1),
            ego_future_location_predicted[:, 1:],
            reduction="none",
        ).sum(
            2
        )  # B x S

        target_progress = (di / d0).mean()

        target_remainder = (di_ / d0).mean()

        # ego_state_mse = F.l1_loss(
        #     ego_future_location_predicted, ego_future_location, reduction="sum") / (
        #     B * S_future)
        # ego_state_mse += F.l1_loss(torch.cos(ego_future_yaw_predicted), torch.cos(
        #     ego_future_yaw), reduction="sum") / (B * S_future)
        # ego_state_mse += F.l1_loss(torch.sin(ego_future_yaw_predicted), torch.sin(
        #     ego_future_yaw), reduction="sum") / (B * S_future)
        ego_state_mse = F.l1_loss(
            ego_future_speed_predicted, ego_future_speed, reduction="sum"
        ) / (B * S_future)

        loss = (
            road_cost * self.cost_weight["road_cost_weight"]
            + road_on_cost * self.cost_weight["road_on_cost_weight"]
            + road_off_cost * self.cost_weight["road_off_cost_weight"]
            + road_red_yellow_cost * self.cost_weight["road_red_yellow_cost_weight"]
            + road_green_cost * self.cost_weight["road_green_cost_weight"]
            + lane_cost * self.cost_weight["lane_cost_weight"]
            + vehicle_cost * self.cost_weight["vehicle_cost_weight"]
            + offroad_cost * self.cost_weight["offroad_cost_weight"]
            + action_mse * self.cost_weight["action_mse_weight"]
            + action_jerk * self.cost_weight["action_jerk_weight"]
            + target_progress * self.cost_weight["target_progress_weight"]
            + target_remainder * self.cost_weight["target_remainder_weight"]
            + ego_state_mse * self.cost_weight["ego_state_mse_weight"]
        )

        if mode == "train":
            self.train_step += B
            step = self.train_step
        elif mode == "val":
            self.val_step += B
            step = self.val_step

        loss_dict = {
            "step": step,
            "road_cost": road_cost,
            "road_on_cost": road_on_cost,
            "road_off_cost": road_off_cost,
            "road_red_yellow_cost": road_red_yellow_cost,
            "road_green_cost": road_green_cost,
            "lane_cost": lane_cost,
            "vehicle_cost": vehicle_cost,
            "offroad_cost": offroad_cost,
            "action_mse": action_mse,
            "action_jerk": action_jerk,
            "target_progress": target_progress,
            "target_remainder": target_remainder,
            "ego_state_mse": ego_state_mse,
            "loss": loss,
        }

        self.render(
            i,
            world_future_bev_,
            cost,
            ego_future_action_predicted,
            ego_future_action[..., :2],
        )

        return loss_dict

    def learn(self, run=None):

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch = epoch
            self.train(epoch, run)
            loss_dict = self.validate(run)
            logger.info(f"Epoch: {epoch}")
            for (k, v) in loss_dict.items():
                logger.info(f"{k}: {v}")
            logger.info("=========================================")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if ((epoch + 1) % self.save_interval == 0) and self.save_path is not None:

                torch.save(
                    {
                        "model_state_dict": self.model.module.get_policy_model().state_dict(),
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
            self._init_canvas()
            x1 = 0
            y1 = 0
            for k in range(self.dataloader_train.batch_size):
                for m in range(self.num_time_step_future - 1):
                    bev = world_future_bev_predicted[k, m + 1]
                    bev[bev > 0.5] = 1
                    bev[bev <= 0.5] = 0
                    bev = bev.detach().cpu().numpy()

                    mask_car = cost["mask_car"][k, m]
                    mask_car = mask_car.detach().cpu().numpy()
                    mask_car = (
                        (
                            (mask_car - mask_car.min())
                            / (mask_car.max() - mask_car.min())
                        )
                        * 255
                    ).astype(np.uint8)

                    mask_car = cv2.applyColorMap(mask_car, cv2.COLORMAP_JET)

                    bev = cv2.cvtColor(
                        BirdViewProducer.as_rgb_with_indices(
                            np.transpose(bev, (1, 2, 0)), [0, 1, 2, 3, 4, 5, 6, 11]
                        ),
                        cv2.COLOR_BGR2RGB,
                    )

                    x2 = x1 + bev.shape[1]
                    y2 = y1 + bev.shape[0]
                    self.canvas_car[y1:y2, x1:x2] = cv2.addWeighted(
                        bev, 0.5, mask_car, 0.5, 0
                    )

                    # Draw ground truth action to the left corner of each bev
                    # as vector
                    action = action_gt[k, m + 1]
                    action = action.detach().cpu().numpy()
                    action = action * 50
                    action = action.astype(np.int32)
                    cv2.arrowedLine(
                        self.canvas_car,
                        (x1 + 50, y1 + 50),
                        (x1 + 50 + action[1], y1 + 50 - action[0]),
                        (255, 255, 255),
                        1,
                        tipLength=0.5,
                    )

                    # Draw predicted action to the left corner of each bev
                    # as vector

                    action = action_pred[k, m + 1]
                    action = action.detach().cpu().numpy()
                    action = action * 50
                    action = action.astype(np.int32)
                    cv2.arrowedLine(
                        self.canvas_car,
                        (x1 + 50, y1 + 50),
                        (x1 + 50 + action[1], y1 + 50 - action[0]),
                        (0, 255, 255),
                        1,
                        tipLength=0.5,
                    )

                    x1 = x2 + 20

                x1 = 0
                y1 = y2 + 20

            x1 = 10
            yy = y1

            for (k, cost_) in enumerate(cost["cost"]):
                cv2.putText(
                    self.canvas_car,
                    f"{k}: {cost_}",
                    (x1, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                yy += 30

            # Save the canvas

            path_car = (
                self.debug_path
                / ("training" if self.model.training else "validation")
                / str(self.epoch)
            )

            path_car.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(path_car / f"{i}_car.png"), self.canvas_car)

    def _init_canvas(self):

        width = (self.num_time_step_future - 1) * (self.bev_width + 20)
        height = self.dataloader_train.batch_size * (self.bev_height + 20) + 300
        self.canvas_car = np.zeros((height, width, 3), dtype=np.uint8)
