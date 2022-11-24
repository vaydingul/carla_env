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
            evaluation_scheme,
            num_time_step_previous=10,
            num_time_step_predict=10,
            threshold=0.25,
            save_path=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.evaluation_scheme = evaluation_scheme
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_predict = num_time_step_predict
        self.threshold = threshold
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

            world_previous_bev = data["bev"]["bev"][:, :self.num_time_step_previous].to(
                self.device).clone()
            world_future_bev = data["bev"]["bev"][:, self.num_time_step_previous:].to(
                self.device).clone()

            for _ in range(self.num_time_step_predict):

                # Predict the future bev
                world_future_bev_predicted = self.model(
                    world_previous_bev, sample_latent=True)

                world_future_bev_predicted = torch.sigmoid(
                    world_future_bev_predicted)
                world_future_bev_predicted[world_future_bev_predicted >
                                           self.threshold] = 1
                world_future_bev_predicted[world_future_bev_predicted <=
                                           self.threshold] = 0

                #  Append the predicted future bev to the list
                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)

                # Update the previous bev
                world_previous_bev = torch.cat(
                    (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

            self._init_canvas(world_previous_bev.shape[0])
            self._draw(data, world_future_bev_predicted_list)

            if render:

                canvas_scaled_half = cv2.resize(
                    self.canvas, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Evaluation", canvas_scaled_half)

            if save:

                self._save(i)

    def _init_canvas(self, num_canvas):

        canvas = np.zeros((self.dataloader.dataset[0]["bev"]["bev"].shape[-2] * 2 + 200, self.dataloader.dataset[0][
            "bev"]["bev"].shape[-1] * (self.num_time_step_previous + self.num_time_step_predict), 3), dtype=np.uint8)
        self.canvas_list = [np.copy(canvas) for _ in range(num_canvas)]

    def _draw(self, data, world_future_bev_predicted_list):

        for (canvas_num, canvas) in enumerate(self.canvas_list):

            self.canvas = canvas
            # Draw the previous bev
            for j in range(self.num_time_step_previous):
                self.canvas[:data["bev"]["bev"].shape[-2], j * data["bev"]["bev"].shape[-1]:(j + 1) * data["bev"]["bev"].shape[-1]] = cv2.cvtColor(
                    self._bev_to_rgb(data["bev"]["bev"][canvas_num, j].detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top-middle of the image
                cv2.putText(self.canvas,
                            f"GT t - {self.num_time_step_previous - j -1}",
                            (data["bev"]["bev"].shape[-1] * j + 10,
                             20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,
                             255,
                             255),
                            2,
                            cv2.LINE_AA)
            # Draw the predicted future bev
            for j in range(self.num_time_step_predict):
                self.canvas[:data["bev"]["bev"].shape[-2], (j + self.num_time_step_previous) * data["bev"]["bev"].shape[-1]:(j + self.num_time_step_previous + 1)
                            * data["bev"]["bev"].shape[-1]] = cv2.cvtColor(self._bev_to_rgb(world_future_bev_predicted_list[j][canvas_num].detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top middle of the image
                cv2.putText(self.canvas,
                            f"P t + {j + 1}",
                            (data["bev"]["bev"].shape[-1] * (j + self.num_time_step_previous) + 10,
                             20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,
                             255,
                             255),
                            2,
                            cv2.LINE_AA)

            # Draw the ground truth future bev below line
            for j in range(
                    self.num_time_step_previous,
                    self.num_time_step_previous +
                    self.num_time_step_predict):
                self.canvas[data["bev"]["bev"].shape[-2] + 200:, (j) * data["bev"]["bev"].shape[-1]:(j + 1) * data["bev"]["bev"].shape[-1]] = cv2.cvtColor(
                    self._bev_to_rgb(data["bev"]["bev"][canvas_num, j].detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top middle of the image
                cv2.putText(self.canvas,
                            f"GT t + {j + 1 - self.num_time_step_previous}",
                            (data["bev"]["bev"].shape[-1] * (j) + 10,
                             data["bev"]["bev"].shape[-2] + 200 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,
                             255,
                             255),
                            2,
                            cv2.LINE_AA)

    def _bev_to_rgb(self, bev):

        # Transpose the bev representation
        bev = bev.transpose(1, 2, 0)

        rgb_image = BirdViewProducer.as_rgb_model(bev)

        return rgb_image

    def _save(self, step):

        if self.save_path is not None:

            for k in range(len(self.canvas_list)):
                cv2.imwrite(
                    f"{self.save_path}/{k + (step * len(self.canvas_list))}.png",
                    self.canvas_list[k])
