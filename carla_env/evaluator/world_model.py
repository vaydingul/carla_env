import torch
import cv2
import numpy as np
from carla_env.bev import BirdViewProducer
import os


class Evaluator(object):

    def __init__(
            self,
            model,
            dataloader,
            device,
            num_time_step_predict,
            save_path=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_time_step_predict = num_time_step_predict
        self.save_path = save_path

        # Create folder at save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.model.to(self.device)

    def evaluate(self, render=True, save=True):

        self.model.eval()

        for i, (data) in enumerate(self.dataloader):

            world_future_bev_predicted_list = []

            world_previous_bev = data["bev"][:, :-1].to(self.device).clone()

            for _ in range(self.num_time_step_predict):

                # Predict the future bev
                world_future_bev_predicted = self.model(
                    world_previous_bev, sample_latent=True)
                world_future_bev_predicted = torch.nn.functional.softmax(
                    world_future_bev_predicted)
                world_future_bev_predicted_max_indices = torch.argmax(
                    world_future_bev_predicted, dim=1)
                world_future_bev_predicted = torch.nn.functional.one_hot(
                    world_future_bev_predicted_max_indices,
                    num_classes=world_future_bev_predicted.shape[1]).permute(
                    0,
                    3,
                    1,
                    2)
                #  Append the predicted future bev to the list
                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)

                # Update the previous bev
                world_previous_bev = torch.cat(
                    (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

            self._init_canvas()
            self._draw(data, world_future_bev_predicted_list)

            if render:

                canvas_scaled_half = cv2.resize(
                    self.canvas, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Evaluation", canvas_scaled_half)

            if save:

                self._save(i)

    def _init_canvas(self):

        self.canvas = np.zeros((self.dataloader.dataset[0]["bev"].shape[-2] * 2 + 200, self.dataloader.dataset[0]["bev"].shape[-1] * (
            self.dataloader.dataset.sequence_length + self.num_time_step_predict), 3), dtype=np.uint8)

    def _draw(self, data, world_future_bev_predicted_list):

        # Draw the previous bev
        for j in range(self.dataloader.dataset.sequence_length - 1):
            self.canvas[:data["bev"].shape[-2], j * data["bev"].shape[-1]:(
                j + 1) * data["bev"].shape[-1]] = self._bev_to_rgb(data["bev"][0, j].detach().cpu().numpy())

        # Draw the future bev
        for j in range(self.num_time_step_predict):
            self.canvas[:data["bev"].shape[-2], (self.dataloader.dataset.sequence_length - 1 + j) * data["bev"].shape[-1]:(
                self.dataloader.dataset.sequence_length + j) * data["bev"].shape[-1]] = self._bev_to_rgb(world_future_bev_predicted_list[j][0].detach().cpu().numpy())

        # Draw the ground truth future bev below line
        for j in range(data["bev"].shape[0]):
            self.canvas[data["bev"].shape[-2] + 200:, (self.dataloader.dataset.sequence_length - 1 + j) * data["bev"].shape[-1]:(
                self.dataloader.dataset.sequence_length + j) * data["bev"].shape[-1]] = self._bev_to_rgb(data["bev"][j, -1].detach().cpu().numpy())

    def _bev_to_rgb(self, bev):

        # Transpose the bev representation
        bev = bev.transpose(1, 2, 0)

        rgb_image = BirdViewProducer.as_rgb_model(bev)

        return rgb_image

    def _save(self, step):

        if self.save_path is not None:

            cv2.imwrite(f"{self.save_path}/{step}.png", self.canvas)
