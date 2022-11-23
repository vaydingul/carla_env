from carla_env.bev import BirdViewProducer
import logging
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from pathlib import Path
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            device,
            cost,
            cost_weight=None,
            num_time_step_previous=10,
            num_time_step_future=10,
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
            save_interval=5):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.device = device
        self.cost = cost
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
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

        self.model.to(self.device)

        self.cost_weight = {}
        self.cost_weight["lane_cost_weight"] = 0.005
        self.cost_weight["vehicle_cost_weight"] = 0.005
        self.cost_weight["red_light_cost_weight"] = 0.005
        self.cost_weight["yellow_light_cost_weight"] = 0.005
        self.cost_weight["offroad_cost_weight"] = 0.005
        self.cost_weight["pedestrian_cost_weight"] = 0.005
        self.cost_weight["offroad_cost_weight"] = 0.005
        self.cost_weight["action_mse_weight"] = 0.005
        self.cost_weight["action_jerk_weight"] = 0.005

        if cost_weight is not None:
            for (k, v) in cost_weight.items():
                self.cost_weight[k] = v

    def train(self, run):

        self.model.train()

        for i, (data) in enumerate(self.dataloader_train):

            world_previous_bev = data["bev"]["bev"][:,
                                                    :self.num_time_step_previous].to(self.device)
            world_future_bev = data["bev"]["bev"][:, self.num_time_step_previous:
                                                  self.num_time_step_previous + self.num_time_step_future].to(self.device)

            world_future_bev_predicted_list = []

            ego_previous_location = data["ego"]["location_array"][:,
                                                                  self.num_time_step_previous - 1, 0:2].to(self.device)

            ego_future_location = data["ego"]["location_array"][:,
                                                                self.num_time_step_previous: self.num_time_step_previous + self.num_time_step_future,
                                                                0:2].to(self.device)

            ego_future_location_predicted_list = []

            ego_previous_yaw = torch.deg2rad(
                data["ego"]["rotation_array"][:, self.num_time_step_previous - 1, 2:].to(self.device))
            ego_future_yaw = torch.deg2rad(data["ego"]["rotation_array"][:,
                                                                         self.num_time_step_previous: self.num_time_step_previous + self.num_time_step_future,
                                                                         2:].to(self.device))
            ego_future_yaw_predicted_list = []

            ego_previous_speed = data["ego"]["velocity_array"][:,
                                                               self.num_time_step_previous - 1].norm(2, -1, keepdim=True).to(self.device)
            ego_future_speed = data["ego"]["velocity_array"][:, self.num_time_step_previous: self.num_time_step_previous +
                                                             self.num_time_step_future].norm(2, -1, keepdim=True).to(self.device)
            ego_future_speed_predicted_list = []

            ego_future_action = data["ego"]["control_array"][:, self.num_time_step_previous:
                                                             self.num_time_step_previous + self.num_time_step_future].to(self.device)
            ego_future_action_predicted_list = []

            command = data["navigation"]["command"][:,
                                                    self.num_time_step_previous - 1].to(self.device)
            command = F.one_hot(command - 1, num_classes=6).float()
            target_location = data["navigation"]["waypoint"][:,
                                                             self.num_time_step_previous - 1, 0:2].to(self.device)

            ego_state_previous = {
                "location": ego_previous_location,
                "yaw": ego_previous_yaw,
                "speed": ego_previous_speed}

            for k in range(self.num_time_step_future):

                # Predict the future bev
                output = self.model(
                    ego_state_previous,
                    world_previous_bev,
                    command,
                    target_location)

                ego_state_next = output["ego_state_next"]
                world_state_next = output["world_state_next"]
                action = output["action"]

                world_future_bev_predicted = torch.sigmoid(
                    world_state_next)

                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)
                ego_future_location_predicted_list.append(
                    ego_state_next["location"])
                ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
                ego_future_speed_predicted_list.append(ego_state_next["speed"])
                ego_future_action_predicted_list.append(action)
                # Predict the future ego location

                # Update the previous bev
                world_previous_bev = torch.cat(
                    (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

                ego_state_previous = ego_state_next

            world_future_bev_predicted = torch.stack(
                world_future_bev_predicted_list, dim=1)

            ego_future_location_predicted = torch.stack(
                ego_future_location_predicted_list, dim=1)

            ego_future_yaw_predicted = torch.stack(
                ego_future_yaw_predicted_list, dim=1)

            ego_future_speed_predicted = torch.stack(
                ego_future_speed_predicted_list, dim=1)

            ego_future_action_predicted = torch.stack(
                ego_future_action_predicted_list, dim=1)

            cost = self.cost(
                ego_future_location_predicted,
                ego_future_yaw_predicted,
                ego_future_speed_predicted,
                world_future_bev_predicted,
            )

            # world_future_bev.requires_grad = True
            # cost = self.cost(
            #     ego_future_location,
            #     ego_future_yaw,
            #     ego_future_speed,
            #     world_future_bev,
            # )

            lane_cost = cost["lane_cost"]
            vehicle_cost = cost["vehicle_cost"]
            green_light_cost = cost["green_light_cost"]
            yellow_light_cost = cost["yellow_light_cost"]
            red_light_cost = cost["red_light_cost"]
            pedestrian_cost = cost["pedestrian_cost"]
            offroad_cost = cost["offroad_cost"]

            ego_future_action[..., 0] -= ego_future_action[..., -1]

            action_mse = F.mse_loss(
                ego_future_action_predicted, ego_future_action[..., :2], reduction="sum")

            action_jerk = torch.diff(
                ego_future_action_predicted,
                dim=1).square().sum()

            target_mse = F.mse_loss(
                ego_previous_location, target_location, reduction="sum")

            loss = lane_cost * self.cost_weight["lane_cost_weight"] + \
                vehicle_cost * self.cost_weight["vehicle_cost_weight"] + \
                green_light_cost * self.cost_weight["green_light_cost_weight"] + \
                yellow_light_cost * self.cost_weight["yellow_light_cost_weight"] + \
                red_light_cost * self.cost_weight["red_light_cost_weight"] + \
                pedestrian_cost * self.cost_weight["pedestrian_cost_weight"] + \
                offroad_cost * self.cost_weight["offroad_cost_weight"] + \
                action_mse * self.cost_weight["action_mse_weight"] + \
                action_jerk * self.cost_weight["action_jerk_weight"] + \
                target_mse * self.cost_weight["target_mse_weight"]

            loss /= world_previous_bev.shape[0] * \
                world_previous_bev.shape[1]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Clip the gradients
            if self.gradient_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_value)
            else:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip_value)

            self.optimizer.step()

            self.train_step += world_previous_bev.shape[0]

            if run is not None:
                run.log({"train/step": self.train_step,
                         "train/lane_cost": lane_cost,
                         "train/vehicle_cost": vehicle_cost,
                         "train/green_light_cost": green_light_cost,
                         "train/yellow_light_cost": yellow_light_cost,
                         "train/red_light_cost": red_light_cost,
                         "train/pedestrian_cost": pedestrian_cost,
                         "train/offroad_cost": offroad_cost,
                         "train/action_mse": action_mse,
                         "train/action_jerk": action_jerk,
                         "train/target_mse": target_mse,
                         "train/loss": loss})

            self.render(i, world_future_bev_predicted, cost)

    def validate(self, run=None):

        self.model.eval()

        lane_cost_list = []
        vehicle_cost_list = []
        red_light_cost_list = []
        yellow_light_cost_list = []
        green_light_cost_list = []
        pedestrian_cost_list = []
        offroad_cost_list = []
        action_mse_list = []
        action_jerk_list = []
        target_mse_list = []
        loss_list = []

        with torch.no_grad():
            for i, (data) in enumerate(self.dataloader_val):

                world_previous_bev = data["bev"]["bev"][:,
                                                        :self.num_time_step_previous].to(self.device)
                world_future_bev = data["bev"]["bev"][:, self.num_time_step_previous:
                                                      self.num_time_step_previous + self.num_time_step_future].to(self.device)

                world_future_bev_predicted_list = []

                ego_previous_location = data["ego"]["location_array"][:,
                                                                      self.num_time_step_previous - 1, 0:2].to(self.device)
                ego_future_location = data["ego"]["location_array"][:,
                                                                    self.num_time_step_previous: self.num_time_step_previous + self.num_time_step_future,
                                                                    0:2].to(self.device)
                ego_future_location_predicted_list = []

                ego_previous_yaw = torch.deg2rad(
                    data["ego"]["rotation_array"][:, self.num_time_step_previous - 1, 1:2].to(self.device))
                ego_future_yaw = torch.deg2rad(data["ego"]["rotation_array"][:,
                                                                             self.num_time_step_previous: self.num_time_step_previous + self.num_time_step_future,
                                                                             1:2].to(self.device))
                ego_future_yaw_predicted_list = []

                ego_previous_speed = data["ego"]["velocity_array"][:,
                                                                   self.num_time_step_previous - 1].norm(2, -1, keepdim=True).to(self.device)
                ego_future_speed = data["ego"]["velocity_array"][:, self.num_time_step_previous: self.num_time_step_previous +
                                                                 self.num_time_step_future].norm(2, -1, keepdim=True).to(self.device)
                ego_future_speed_predicted_list = []

                ego_future_action = data["ego"]["control_array"][:, self.num_time_step_previous:
                                                                 self.num_time_step_previous + self.num_time_step_future].to(self.device)
                ego_future_action_predicted_list = []

                command = data["navigation"]["command"][:,
                                                        self.num_time_step_previous - 1].to(self.device)
                command = F.one_hot(command - 1, num_classes=6).float()
                target_location = data["navigation"]["waypoint"][:,
                                                                 self.num_time_step_previous - 1, 0:2].to(self.device)

                ego_state_previous = {
                    "location": ego_previous_location,
                    "yaw": ego_previous_yaw,
                    "speed": ego_previous_speed}

                for k in range(self.num_time_step_future):

                    # Predict the future bev
                    output = self.model(
                        ego_state_previous,
                        world_previous_bev,
                        command,
                        target_location)

                    ego_state_next = output["ego_state_next"]
                    world_state_next = output["world_state_next"]
                    action = output["action"]

                    world_future_bev_predicted = torch.sigmoid(
                        world_state_next)

                    world_future_bev_predicted_list.append(
                        world_future_bev_predicted)
                    ego_future_location_predicted_list.append(
                        ego_state_next["location"])
                    ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
                    ego_future_speed_predicted_list.append(
                        ego_state_next["speed"])
                    ego_future_action_predicted_list.append(action)
                    # Predict the future ego location

                    # Update the previous bev
                    world_previous_bev = torch.cat(
                        (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

                    ego_state_previous = ego_state_next

                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1)

                ego_future_location_predicted = torch.stack(
                    ego_future_location_predicted_list, dim=1)

                ego_future_yaw_predicted = torch.stack(
                    ego_future_yaw_predicted_list, dim=1)

                ego_future_speed_predicted = torch.stack(
                    ego_future_speed_predicted_list, dim=1)

                ego_future_action_predicted = torch.stack(
                    ego_future_action_predicted_list, dim=1)

                cost = self.cost(
                    ego_future_location_predicted,
                    ego_future_yaw_predicted,
                    ego_future_speed_predicted,
                    world_future_bev_predicted,
                )

                # cost = self.cost(
                #     ego_future_location,
                #     ego_future_yaw,
                #     ego_future_speed,
                #     world_future_bev,
                # )

                lane_cost = cost["lane_cost"]
                vehicle_cost = cost["vehicle_cost"]
                green_light_cost = cost["green_light_cost"]
                yellow_light_cost = cost["yellow_light_cost"]
                red_light_cost = cost["red_light_cost"]
                pedestrian_cost = cost["pedestrian_cost"]
                offroad_cost = cost["offroad_cost"]

                ego_future_action[..., 0] -= ego_future_action[..., -1]

                action_mse = F.mse_loss(
                    ego_future_action_predicted, ego_future_action[..., :2], reduction="sum")

                action_jerk = torch.diff(
                    ego_future_action_predicted,
                    dim=1).square().sum()

                target_mse = F.mse_loss(
                    ego_previous_location, target_location, reduction="sum")

                loss = lane_cost * self.cost_weight["lane_cost_weight"] + \
                    vehicle_cost * self.cost_weight["vehicle_cost_weight"] + \
                    green_light_cost * self.cost_weight["green_light_cost_weight"] + \
                    yellow_light_cost * self.cost_weight["yellow_light_cost_weight"] + \
                    red_light_cost * self.cost_weight["red_light_cost_weight"] + \
                    pedestrian_cost * self.cost_weight["pedestrian_cost_weight"] + \
                    offroad_cost * self.cost_weight["offroad_cost_weight"] + \
                    action_mse * self.cost_weight["action_mse_weight"] + \
                    action_jerk * self.cost_weight["action_jerk_weight"] + \
                    target_mse * self.cost_weight["target_mse_weight"]

                loss /= world_previous_bev.shape[0] * \
                    world_previous_bev.shape[1]

                lane_cost_list.append(lane_cost.item())
                vehicle_cost_list.append(vehicle_cost.item())
                green_light_cost_list.append(green_light_cost.item())
                yellow_light_cost_list.append(yellow_light_cost.item())
                red_light_cost_list.append(red_light_cost.item())
                pedestrian_cost_list.append(pedestrian_cost.item())
                offroad_cost_list.append(offroad_cost.item())
                action_mse_list.append(action_mse.item())
                action_jerk_list.append(action_jerk.item())
                target_mse_list.append(target_mse.item())
                loss_list.append(loss.item())

                self.val_step += world_previous_bev.shape[0]

                self.render(i, world_future_bev_predicted, cost)

            lane_cost_mean = np.mean(lane_cost_list)
            vehicle_cost_mean = np.mean(vehicle_cost_list)
            green_light_cost_mean = np.mean(green_light_cost_list)
            yellow_light_cost_mean = np.mean(yellow_light_cost_list)
            red_light_cost_mean = np.mean(red_light_cost_list)
            pedestrian_cost_mean = np.mean(pedestrian_cost_list)
            offroad_cost_mean = np.mean(offroad_cost_list)
            action_mse_mean = np.mean(action_mse_list)
            action_jerk_mean = np.mean(action_jerk_list)
            target_mse_mean = np.mean(target_mse_list)
            loss_mean = np.mean(loss_list)

            if run is not None:
                run.log({"val/step": self.val_step,
                         "val/lane_cost": lane_cost_mean,
                         "val/vehicle_cost": vehicle_cost_mean,
                         "val/green_light_cost": green_light_cost_mean,
                         "val/yellow_light_cost": yellow_light_cost_mean,
                         "val/red_light_cost": red_light_cost_mean,
                         "val/pedestrian_cost": pedestrian_cost_mean,
                         "val/offroad_cost": offroad_cost_mean,
                         "val/action_mse": action_mse_mean,
                         "val/action_jerk": action_jerk_mean,
                         "val/target_mse": target_mse_mean,
                         "val/loss": loss_mean})
                if self.lr_scheduler is not None:
                    run.log({"val/lr": self.lr_scheduler.get_last_lr()[0]})

            return (
                lane_cost_mean,
                vehicle_cost_mean,
                offroad_cost_mean,
                action_mse_mean,
                loss_mean)

    def learn(self, run=None):

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch = epoch
            self.train(run)
            (lane_cost,
             vehicle_cost,
             offroad_cost,
             action_mse,
             loss) = self.validate(run)
            logger.info(
                f"Epoch: {epoch}, Lane Cost: {lane_cost}, Vehicle Cost: {vehicle_cost}, Offroad Cost: {offroad_cost}, Action MSE: {action_mse}, Loss: {loss}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if ((epoch + 1) % self.save_interval ==
                    0) and self.save_path is not None:

                torch.save({
                    "model_state_dict": self.model.get_policy_model().state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "epoch": epoch,
                    "train_step": self.train_step,
                    "val_step": self.val_step},
                    self.save_path /
                    Path(f"checkpoint_{epoch}.pt"))

                if run is not None:
                    run.save(str(self.save_path /
                                 Path(f"checkpoint_{epoch}.pt")))

    def render(self, i, world_future_bev_predicted, cost):

        if self.debug_render:
            self._init_canvas()
            x1 = 0
            y1 = 0
            for k in range(self.dataloader_train.batch_size):
                for m in range(self.num_time_step_future - 1):
                    bev = world_future_bev_predicted[k, m]
                    bev[bev > 0.5] = 1
                    bev[bev <= 0.5] = 0
                    bev = bev.detach().cpu().numpy()

                    mask_car = cost["mask_car"][k, m]
                    mask_car = mask_car.detach().cpu().numpy()
                    mask_car = (((mask_car -
                                  mask_car.min()) /
                                 (mask_car.max() -
                                  mask_car.min())) *
                                255).astype(np.uint8)

                    mask_side = cost["mask_side"][k, m]
                    mask_side = mask_side.detach().cpu().numpy()
                    mask_side = (((mask_side -
                                   mask_side.min()) /
                                  (mask_side.max() -
                                 mask_side.min())) *
                                 255).astype(np.uint8)

                    mask_car = cv2.applyColorMap(mask_car, cv2.COLORMAP_JET)
                    mask_side = cv2.applyColorMap(mask_side, cv2.COLORMAP_JET)

                    bev = cv2.cvtColor(
                        BirdViewProducer.as_rgb_model(
                            np.transpose(
                                bev, (1, 2, 0))), cv2.COLOR_BGR2RGB)

                    x2 = x1 + bev.shape[1]
                    y2 = y1 + bev.shape[0]
                    self.canvas_car[y1:y2, x1:x2] = cv2.addWeighted(
                        bev, 0.5, mask_car, 0.5, 0)
                    self.canvas_side[y1:y2, x1:x2] = cv2.addWeighted(
                        bev, 0.5, mask_side, 0.5, 0)
                    x1 = x2 + 20

                x1 = 0
                y1 = y2 + 20

            y1 += 15

            cv2.putText(
                self.canvas_car,
                f"lane_cost: {cost['lane_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)
            y1 += 15
            cv2.putText(
                self.canvas_car,
                f"vehicle_cost: {cost['vehicle_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)
            y1 += 15
            cv2.putText(
                self.canvas_car,
                f"offroad_cost: {cost['offroad_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)

            cv2.putText(
                self.canvas_side,
                f"lane_cost: {cost['lane_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)
            y1 += 15
            cv2.putText(
                self.canvas_side,
                f"vehicle_cost: {cost['vehicle_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)
            y1 += 15
            cv2.putText(
                self.canvas_side,
                f"offroad_cost: {cost['offroad_cost']}",
                (x1,
                 y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 255,
                 255),
                1,
                cv2.LINE_AA)

            # Save the canvas
            path_car = Path(
                f"./figures/dfm_km_policy_training_debug/{'training' if self.model.training else 'validation'}/{self.epoch}")
            path_car.mkdir(parents=True, exist_ok=True)

            path_side = Path(
                f"./figures/dfm_km_policy_training_debug/{'training' if self.model.training else 'validation'}/{self.epoch}")
            path_side.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(path_car / f"{i}_car.png"), self.canvas_car)
            cv2.imwrite(str(path_side / f"{i}_side.png"), self.canvas_side)

    def _init_canvas(self):

        width = (self.num_time_step_future - 1) * (self.bev_width + 20)
        height = self.dataloader_train.batch_size * \
            (self.bev_height + 20) + 100
        self.canvas_car = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas_side = np.zeros((height, width, 3), dtype=np.uint8)
