import torch
from torch import nn
import numpy as np
import cv2
from carla_env.bev import BirdViewProducer


class ModelPredictiveControl(nn.Module):
    """ModelPredictiveControl controller."""

    def __init__(
            self,
            device,
            batch_size,
            rollout_length,
            action_size,
            number_of_optimization_iterations,
            cost,
            ego_model,
            init_action="zeros",
            world_model=None,
            render_cost=False):
        """Initialize."""

        super(ModelPredictiveControl, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.action_size = action_size
        self.number_of_optimization_iterations = number_of_optimization_iterations
        self.ego_model = ego_model
        self.init_action = init_action
        self.world_model = world_model
        self.cost = cost
        self.render_cost = render_cost

        self.reset()

        if self.render_cost:

            self._initialize_rendering()

    def forward_world_model(self, bev):

        bev_predicted = []
        # bev_predicted.append(bev[:, -1].unsqueeze(1))

        for i in range(self.rollout_length):

            if self.world_model is not None:

                bev_ = self.world_model(bev, sample_latent=True)
                bev_ = torch.sigmoid(bev_)
                #bev_[bev_ > 0.5] = 1
                #bev_[bev_ <= 0.5] = 0

                bev = torch.cat([bev[:, 1:], bev_.unsqueeze(1)], dim=1)

                bev_predicted.append(bev_.unsqueeze(1))

            else:

                bev_predicted.append(bev.unsqueeze(1))

        bev_predicted = torch.cat(bev_predicted, dim=1)

        return bev_predicted.clone()

    def forward_ego_model(self, location, yaw, speed):
        """Run a single step of ModelPredictiveControl."""

        location_predicted = []
        yaw_predicted = []
        speed_predicted = []

        location_predicted.append(location)
        yaw_predicted.append(yaw)
        speed_predicted.append(speed)

        ego_state = {"location": location,
                     "yaw": yaw,
                     "speed": speed}

        for i in range(self.rollout_length - 1):

            action_ = self.action[:, i:i+1].clone()

            ego_state_next = self.ego_model(
                ego_state, action_)

            location_predicted.append(ego_state_next["location"])
            yaw_predicted.append(ego_state_next["yaw"])
            speed_predicted.append(ego_state_next["speed"])

            ego_state = ego_state_next

        location_predicted = torch.cat(location_predicted, dim=1)
        yaw_predicted = torch.cat(yaw_predicted, dim=1)
        speed_predicted = torch.cat(speed_predicted, dim=1)

        return (
            location_predicted.clone(),
            yaw_predicted.clone(),
            speed_predicted.clone()
        )

    def step(self, initial_state, target_state, bev):
        """Optimize the action."""

        bev_predicted = self.forward_world_model(bev.clone())

        for k in range(self.number_of_optimization_iterations):

            self.optimizer.zero_grad()

            location = initial_state[:, :, 0:2].clone()
            yaw = initial_state[:, :, 2:3].clone()
            speed = initial_state[:, :, 3:4].clone()

            (location_predicted,
             yaw_predicted,
             speed_predicted
             ) = self.forward_ego_model(location=location,
                                        yaw=yaw,
                                        speed=speed
                                        )

            cost = self._calculate_cost(
                predicted_location=location_predicted.clone(),
                predicted_yaw=yaw_predicted.clone(),
                predicted_speed=speed_predicted.clone(),
                predicted_bev=bev_predicted.clone(),
                target_state=target_state.clone(),
                last_step=k == self.number_of_optimization_iterations - 1)

            cost.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.action, 1)

            self.optimizer.step()

        return (
            self.action.detach().cpu().numpy(),
            location_predicted[0].detach().cpu().numpy(),
            cost.item(),
            self.canvas)

    def reset(self, initial_guess=None):
        """Reset the controller."""

        if initial_guess is None:

            if self.init_action == "zeros":

                action = torch.zeros(
                    (self.batch_size,
                     self.rollout_length,
                     self.action_size),
                    device=self.device,
                    dtype=torch.float32)

            elif self.init_action == "random":

                action = torch.randn(
                    (self.batch_size,
                     self.rollout_length,
                     self.action_size),
                    device=self.device,
                    dtype=torch.float32)

            elif self.init_action == "ones":

                action = torch.ones(
                    (self.batch_size,
                     self.rollout_length,
                     self.action_size),
                    device=self.device,
                    dtype=torch.float32)

            else:

                raise NotImplementedError

        else:

            action = torch.tensor(
                initial_guess,
                device=self.device,
                dtype=torch.float32).unsqueeze(0)

        self.action = nn.Parameter(action, requires_grad=True)

        self.optimizer = torch.optim.Adam((self.action, ), lr=0.05)

    def _calculate_cost(
            self,
            predicted_location,
            predicted_yaw,
            predicted_speed,
            predicted_bev,
            target_state,
            last_step=False):
        """Calculate the cost."""

        # Organize the dimensions for the cost module
        # predicted_location.squeeze_(0)
        # predicted_rotation.squeeze_(0)
        # predicted_speed.squeeze_(0)
        # target_state.squeeze_(0)
        # predicted_bev.squeeze_(0)
        # Convert bev to torch tensor

        # Calculate the cost
        cost = self.cost(predicted_location,
                         predicted_yaw,
                         predicted_speed,
                         predicted_bev,
                         )

        self.predicted_bev = predicted_bev.clone().detach().cpu().numpy()[0]

        self.lane_cost = cost["lane_cost"]
        self.vehicle_cost = cost["vehicle_cost"]
        # green_light_cost = cost["green_light_cost"]
        # yellow_light_cost = cost["yellow_light_cost"]
        # red_light_cost = cost["red_light_cost"]
        # pedestrian_cost = cost["pedestrian_cost"]
        self.offroad_cost = cost["offroad_cost"]
        self.mask_car = cost["mask_car"][0]
        self.mask_side = cost["mask_side"][0]

        cost = torch.tensor(0.0).to(self.device)
        cost += self.lane_cost / 50
        cost += self.vehicle_cost / 50
        #cost += self.green_light_cost
        #cost += self.yellow_light_cost
        #cost += self.red_light_cost
        #cost += self.pedestrian_cost
        cost += self.offroad_cost / 10

        cost += torch.nn.functional.l1_loss(predicted_location[..., :1], target_state[..., :1].expand(
            *(predicted_location[..., :1].shape))) * 10
        cost += torch.nn.functional.l1_loss(predicted_location[..., 1:2], target_state[..., 1:2].expand(
            *(predicted_location[..., 1:2].shape))) * 10
        cost += torch.nn.functional.l1_loss(torch.cos(predicted_yaw), torch.cos(
            target_state[..., 2:3].expand(*(predicted_yaw.shape))))
        cost += torch.nn.functional.l1_loss(torch.sin(predicted_yaw), torch.sin(
            target_state[..., 2:3].expand(*(predicted_yaw.shape))))
        # cost += torch.nn.functional.l1_loss(predicted_speed,
        #                                     target_state[...,
        #                                                  3:4].expand(*(predicted_speed.shape)))

        cost += torch.diff(self.action[..., 0], dim=1).square().sum()
        cost += torch.diff(self.action[..., 1], dim=1).square().sum()
        # cost += -torch.diff(self.action, dim=0).square().sum() * 0.1
        if self.render_cost and last_step:

            self.render()

        return cost

    def render(self):

        self.canvas = np.zeros_like(self.canvas)

        offset_x = 0
        offset_y = 0

        self.predicted_bev[self.predicted_bev > 0.5] = 1
        self.predicted_bev[self.predicted_bev <= 0.5] = 0

        for k in range(self.mask_car.shape[0]):

            bev_ = cv2.cvtColor(
                BirdViewProducer.as_rgb_with_indices(
                    np.transpose(
                        self.predicted_bev[k], (1, 2, 0)), indices=[
                        0, 5, 6, 8, 9, 9, 10, 11]), cv2.COLOR_BGR2RGB)

            # Draw mask_car side by side
            mask_car_ = self.mask_car[k].detach().cpu().numpy()
            # Normalize to 0-255 int
            mask_car_ = (mask_car_ - mask_car_.min()) / \
                (mask_car_.max() - mask_car_.min()) * 255
            mask_car_ = mask_car_.astype(np.uint8)
            mask_car_color_map = cv2.applyColorMap(mask_car_, cv2.COLORMAP_JET)
            self.canvas[offset_y:offset_y + mask_car_.shape[0],
                        offset_x:offset_x + mask_car_.shape[1],
                        :] = cv2.addWeighted(bev_,
                                             0.5,
                                             mask_car_color_map,
                                             0.5,
                                             0)

            offset_x += mask_car_.shape[1] + 10

        offset_x = 0
        offset_y += mask_car_.shape[0] + 20

        for k in range(self.mask_side.shape[0]):
            bev_ = cv2.cvtColor(
                BirdViewProducer.as_rgb_with_indices(
                    np.transpose(
                        self.predicted_bev[k], (1, 2, 0)), indices=[
                        0, 5, 6, 8, 9, 9, 10, 11]), cv2.COLOR_BGR2RGB)
            # Draw mask_car side by side
            mask_side_ = self.mask_side[k].detach().cpu().numpy()
            # Normalize to 0-255 int
            mask_side_ = (mask_side_ - mask_side_.min()) / \
                (mask_side_.max() - mask_side_.min()) * 255
            mask_side_ = mask_side_.astype(np.uint8)
            mask_side_color_map = cv2.applyColorMap(
                mask_side_, cv2.COLORMAP_JET)
            self.canvas[offset_y:offset_y + mask_side_.shape[0],
                        offset_x:offset_x + mask_side_.shape[1],
                        :] = cv2.addWeighted(bev_,
                                             0.5,
                                             mask_side_color_map,
                                             0.5,
                                             0)

            offset_x += mask_side_.shape[1] + 10

        offset_x = 0
        offset_y += mask_side_.shape[0] + 20

        cv2.putText(
            self.canvas,
            f"Offroad Cost: {self.offroad_cost}",
            (offset_x,
             offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255,
             255,
             0),
            1)
        offset_y += 50
        cv2.putText(
            self.canvas,
            f"Vehicle Cost: {self.vehicle_cost}",
            (offset_x,
             offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255,
             255,
             0),
            1)
        offset_y += 50
        cv2.putText(
            self.canvas,
            f"Lane Cost: {self.lane_cost}",
            (offset_x,
             offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255,
             255,
             0),
            1)

        # canvas_display = cv2.resize(self.canvas, (0, 0), fx=0.9, fy=0.9)

    def _initialize_rendering(self):

        self.canvas = np.zeros(
            (self.cost.image_height *
             3,
             self.rollout_length *
             self.cost.image_width,
             3),
            np.uint8)
