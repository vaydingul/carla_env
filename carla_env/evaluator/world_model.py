import torch
import cv2
import numpy as np
from carla_env.bev import BirdViewProducer, BirdViewMasks
import os
from torchmetrics.classification import BinaryJaccardIndex
import pandas as pd


class Evaluator(object):

    def __init__(
            self,
            model,
            dataloader,
            device,
            report_iou=True,
            num_time_step_previous=10,
            num_time_step_predict=10,
            threshold=0.25,
            bev_selected_channels=None,
            save_path=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.report_iou = report_iou
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_predict = num_time_step_predict
        self.threshold = threshold
        self.save_path = save_path
        self.bev_selected_channels = bev_selected_channels
        # Create folder at save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.model.to(self.device)

    def evaluate(self, render=True, save=True):

        self.model.eval()

        if self.report_iou:
            bji = BinaryJaccardIndex(
                compute_on_cpu=True,
                ignore_index=255).to(
                self.device)
            iou_dict_list = []

        for i, (data) in enumerate(self.dataloader):

            world_future_bev_predicted_list = []

            world_previous_bev = data["bev_world"]["bev"][:, :self.num_time_step_previous].to(
                self.device).clone()
            world_future_bev = data["bev_world"]["bev"][:, self.num_time_step_previous:
                                                        self.num_time_step_previous + self.num_time_step_predict].to(self.device).clone()

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

            world_future_bev_predicted = torch.stack(
                world_future_bev_predicted_list, dim=1)

            iou_dict = {}
            if self.report_iou:

                for (ix, channel) in enumerate(self.bev_selected_channels):
                    
                    
                    world_future_bev_predicted_ = world_future_bev_predicted[:, :, ix].clone(
                    ).view(-1, *world_future_bev_predicted.shape[-2:]).to(torch.uint8)
                    world_future_bev_ = world_future_bev[:, :, ix].clone(
                    ).view(-1, *world_future_bev.shape[-2:]).to(torch.uint8)

                    iou = bji(
                        world_future_bev_predicted_,
                        world_future_bev_)

                    iou_dict[f"{BirdViewMasks.bottom_to_top()[channel]}"] = iou.cpu().numpy()
                
                iou_dict_list.append(iou_dict)

            self._init_canvas(world_previous_bev.shape[0])
            self._draw(data, world_future_bev_predicted_list,
                       iou_dict if self.report_iou else None)

            if render:

                canvas_scaled_half = cv2.resize(
                    self.canvas, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Evaluation", canvas_scaled_half)

            if save:

                self._save(i)

        if self.report_iou:
            iou_dict_mean = {}
            for k in iou_dict_list[0].keys():
                iou_dict_mean[k] = np.nanmean(
                    [iou_dict[k] for iou_dict in iou_dict_list])
            # Write to a csv file
            df = pd.DataFrame(iou_dict_mean, index=[0])
            df.to_csv(os.path.join(self.save_path, "iou.csv"))

    def _init_canvas(self, num_canvas):

        canvas = np.zeros((self.dataloader.dataset[0]["bev_world"]["bev"].shape[-2] * 2 + 200, self.dataloader.dataset[0][
            "bev_world"]["bev"].shape[-1] * (self.num_time_step_previous + self.num_time_step_predict), 3), dtype=np.uint8)
        self.canvas_list = [np.copy(canvas) for _ in range(num_canvas)]

    def _draw(self, data, world_future_bev_predicted_list, iou_dict=None):

        for (canvas_num, canvas) in enumerate(self.canvas_list):

            self.canvas = canvas
            # Draw the previous bev
            for j in range(self.num_time_step_previous):
                self.canvas[:data["bev_world"]["bev"].shape[-2], j * data["bev_world"]["bev"].shape[-1]:(j + 1) * data["bev_world"]["bev"].shape[-1]] = cv2.cvtColor(
                    self._bev_to_rgb(data["bev_world"]["bev"][canvas_num, j].clone().detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top-middle of the image
                cv2.putText(self.canvas,
                            f"GT t - {self.num_time_step_previous - j -1}",
                            (data["bev_world"]["bev"].shape[-1] * j + 10,
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
                self.canvas[:data["bev_world"]["bev"].shape[-2], (j + self.num_time_step_previous) * data["bev_world"]["bev"].shape[-1]:(j + self.num_time_step_previous + 1)
                            * data["bev_world"]["bev"].shape[-1]] = cv2.cvtColor(self._bev_to_rgb(world_future_bev_predicted_list[j][canvas_num].clone().detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top middle of the image
                cv2.putText(self.canvas,
                            f"P t + {j + 1}",
                            (data["bev_world"]["bev"].shape[-1] * (j + self.num_time_step_previous) + 10,
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
                self.canvas[data["bev_world"]["bev"].shape[-2] + 200:, (j) * data["bev_world"]["bev"].shape[-1]:(j + 1) * data["bev_world"]["bev"].shape[-1]] = cv2.cvtColor(
                    self._bev_to_rgb(data["bev_world"]["bev"][canvas_num, j].clone().detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
                # Put text on the top middle of the image
                cv2.putText(self.canvas,
                            f"GT t + {j + 1 - self.num_time_step_previous}",
                            (data["bev_world"]["bev"].shape[-1] * (j) + 10,
                             data["bev_world"]["bev"].shape[-2] + 200 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,
                             255,
                             255),
                            2,
                            cv2.LINE_AA)

            if iou_dict is not None:
                for j, (k, v) in enumerate(iou_dict.items()):
                    cv2.putText(self.canvas,
                                f"{k}: {v:.3f}",
                                (10, data["bev_world"]["bev"].shape[-2] + 20 + 20 + (j + 1) * 20),
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

        # Old BEV
        # rgb_image = BirdViewProducer.as_rgb_with_indices(bev, [0, 5, 6, 8, 9, 9, 10, 11])

        # New BEV
        if self.bev_selected_channels is None:
            rgb_image = BirdViewProducer.as_rgb(bev)
        else:
            rgb_image = BirdViewProducer.as_rgb_with_indices(
                bev, self.bev_selected_channels)

        return rgb_image

    def _save(self, step):

        if self.save_path is not None:

            for k in range(len(self.canvas_list)):
                cv2.imwrite(
                    f"{self.save_path}/{k + (step * len(self.canvas_list))}.png",
                    self.canvas_list[k])
