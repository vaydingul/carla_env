import torch
from torch import nn
import numpy as np
import cv2
from carla_env.bev import BirdViewProducer

class MPC(nn.Module):
    """MPC controller."""

    def __init__(
            self,
            device,
            action_size,
            rollout_length,
            number_of_optimization_iterations,
            model,
            cost,
            render_cost=False):
        """Initialize."""

        super(MPC, self).__init__()

        #self.model = model.to(device)
        self.device = device
        self.action_size = action_size
        self.rollout_length = rollout_length
        self.number_of_optimization_iterations = number_of_optimization_iterations
        self.model = model
        self.cost = cost
        self.render_cost = render_cost

        self.action = torch.zeros(
            (1, self.rollout_length, self.action_size), device=self.device)
        #self.action[..., 0] = torch.ones_like(self.action[..., 0])
        self.action = nn.Parameter(self.action, requires_grad=True)
        self.optimizer = torch.optim.SGD((self.action, ), lr=.5)

        if self.render_cost:

            self._initialize_rendering()

    def forward(self, location, rotation, speed):
        """Run a single step of MPC."""

        location_predicted = []
        rotation_predicted = []
        speed_predicted = []

        location_predicted.append(location)
        rotation_predicted.append(rotation)
        speed_predicted.append(speed)

        for i in range(self.rollout_length - 1):

            action_ = self.action[:, i, :].clone()

            (location, rotation, speed) = self.model(
                location, rotation, speed, action_)

            location_predicted.append(location)
            rotation_predicted.append(rotation)
            speed_predicted.append(speed)

        location_predicted = torch.cat(location_predicted, dim=1)
        rotation_predicted = torch.cat(rotation_predicted, dim=1)
        speed_predicted = torch.cat(speed_predicted, dim=1)

        return (
            location_predicted.clone(),
            rotation_predicted.clone(),
            speed_predicted.clone())

    def step(self, initial_state, target_state, bev):
        """Optimize the action."""

        for _ in range(self.number_of_optimization_iterations):

            self.optimizer.zero_grad()

            location = initial_state[:, :, 0:2].clone()
            rotation = initial_state[:, :, 2:3].clone()
            speed = initial_state[:, :, 3:4].clone()

            location_predicted, rotation_predicted, speed_predicted = self.forward(
                location, rotation, speed)

            cost = self._calculate_cost(
                predicted_location=location_predicted.clone(),
                predicted_rotation=rotation_predicted.clone(),
                predicted_speed=speed_predicted.clone(),
                bev=bev,
                target_state=target_state.clone())

            cost.backward(retain_graph=True)

            torch.nn.utils.clip_grad_value_(self.action, 0.5)
            self.optimizer.step()

        return list(self.action[0, 0, :].detach().cpu().numpy(
        )), location_predicted[0].detach().cpu().numpy(), cost.item()

    def reset(self, initial_guess=None):
        """Reset the controller."""
        if initial_guess is None:
            # Reset the action
            # action = torch.randn((1, self.rollout_length, 3), device=self.device)
            # action[..., 2] = torch.randint(0, 2, (1, self.rollout_length), device=self.device, dtype=torch.float32)
            action = torch.zeros(
                (1,
                 self.rollout_length,
                 self.action_size),
                device=self.device,
                dtype=torch.float32)
            #action[..., -1] = torch.randint_like(action[..., -1], 0, 2, dtype=torch.float32)
            self.action = nn.Parameter(action)
        else:
            self.action = nn.Parameter(
                initial_guess.clone().detach().requires_grad_(True).repeat(
                    1, self.rollout_length, 1))

        self.optimizer = torch.optim.Adam((self.action, ), lr=0.05)

    def _calculate_cost(
            self,
            predicted_location,
            predicted_rotation,
            predicted_speed,
            bev,
            target_state):
        """Calculate the cost."""

        # Organize the dimensions for the cost module
        predicted_location.squeeze_(0)
        predicted_rotation.squeeze_(0)
        predicted_speed.squeeze_(0)
        target_state.squeeze_(0)
        # Convert bev to torch tensor
        self.bev = bev
        bev = torch.from_numpy(bev).float().to(
            predicted_location.device).repeat(
            predicted_location.shape[0] - 1, 1, 1, 1)

        # Calculate the cost
        (self.lane_cost,
            self.vehicle_cost,
            self.green_light_cost,
            self.yellow_light_cost,
            self.red_light_cost,
            self.pedestrian_cost,
            self.offroad_cost,
            self.mask_car,
            self.mask_side) = self.cost(predicted_location,
                                        predicted_rotation,
                                        predicted_speed,
                                        bev)

        cost = torch.tensor(0.0).to(self.device)
        cost += self.lane_cost
        #cost += self.vehicle_cost
        #cost += self.green_light_cost
        #cost += self.yellow_light_cost
        #cost += self.red_light_cost
        #cost += self.pedestrian_cost
        cost += self.offroad_cost

        cost /= 250

        cost += torch.maximum(torch.nn.functional.l1_loss(predicted_location[..., :1], target_state[..., :1].expand(
            *(predicted_location[..., :1].shape))) , torch.nn.functional.l1_loss(predicted_location[..., 1:2], target_state[..., 1:2].expand(
            *(predicted_location[..., 1:2].shape))))
        # cost += torch.nn.functional.l1_loss(predicted_location[..., 1:2], target_state[..., 1:2].expand(
        #     *(predicted_location[..., 1:2].shape)))
        cost += torch.nn.functional.l1_loss(
            torch.cos(predicted_rotation), torch.cos(target_state[..., 2:3].expand(*(predicted_rotation.shape))))
        cost += torch.nn.functional.l1_loss(
            torch.sin(predicted_rotation), torch.sin(target_state[..., 2:3].expand(*(predicted_rotation.shape))))
        cost += torch.nn.functional.l1_loss(predicted_speed,
                                            target_state[...,
                                                         3:4].expand(*(predicted_speed.shape)))

        # cost += torch.diff(self.action[..., 0], dim=1).square().sum() * 0.2
        cost += torch.diff(self.action[..., 1], dim=1).square().sum() * 0.8

        if self.render_cost:
            self.render()

        return cost

    def render(self):

        offset_x = 0
        offset_y = 0
        bev_ = cv2.cvtColor(BirdViewProducer.as_rgb(self.bev), cv2.COLOR_BGR2RGB)

        for k in range(self.mask_car.shape[0]):

            # Draw mask_car side by side
            mask_car_ = self.mask_car[k].detach().cpu().numpy()
            # Normalize to 0-255 int
            mask_car_ = (mask_car_ - mask_car_.min()) / \
                (mask_car_.max() - mask_car_.min()) * 255
            mask_car_ = mask_car_.astype(np.uint8)
            mask_car_color_map = cv2.applyColorMap(mask_car_, cv2.COLORMAP_JET)
            self.canvas[offset_y:offset_y + mask_car_.shape[0],
                        offset_x:offset_x + mask_car_.shape[1], :] = cv2.addWeighted(bev_, 0.5, mask_car_color_map, 0.5, 0)

            offset_x += mask_car_.shape[1] + 10

        offset_x = 0
        offset_y += mask_car_.shape[0] + 20

        for k in range(self.mask_side.shape[0]):

            # Draw mask_car side by side
            mask_side_ = self.mask_side[k].detach().cpu().numpy()
            # Normalize to 0-255 int
            mask_side_ = (mask_side_ - mask_side_.min()) / \
                (mask_side_.max() - mask_side_.min()) * 255
            mask_side_ = mask_side_.astype(np.uint8)
            mask_side_color_map = cv2.applyColorMap(mask_side_, cv2.COLORMAP_JET)
            self.canvas[offset_y:offset_y + mask_side_.shape[0],
                        offset_x:offset_x + mask_side_.shape[1], :] = cv2.addWeighted(bev_, 0.5, mask_side_color_map, 0.5, 0)

            offset_x += mask_side_.shape[1] + 10

        canvas_display = cv2.resize(self.canvas, (0, 0), fx=0.9, fy=0.9)

        cv2.imshow("Cost", canvas_display)

    def _initialize_rendering(self):

        self.canvas = np.zeros(
            (self.cost.image_height *
             3,
             self.rollout_length *
             self.cost.image_width,
             3),
            np.uint8)
        cv2.imshow("Cost", self.canvas)

        if cv2.waitKey(1) == ord('q'):

            # press q to terminate the loop
            cv2.destroyAllWindows()
