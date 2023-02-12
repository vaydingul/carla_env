import torch
import cv2
import numpy as np
from carla_env.bev import BirdViewProducer, BirdViewMasks
import os
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
    MultilabelF1Score,
    MultilabelJaccardIndex,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelROC,
    MultilabelStatScores)
import pandas as pd
from utils.plot_utils import plot_roc, plot_confusion_matrix
NUM_LABELS = 8
THRESHOLD = 0.5
NUM_THRESHOLD = 10
AVERAGE = None

SETTING_1 = {
    "threshold": THRESHOLD,
    "average": AVERAGE,
    "num_labels": NUM_LABELS,
    "compute_on_cpu": True
}

SETTING_2 = {
    "thresholds": NUM_THRESHOLD,
    "num_labels": NUM_LABELS,
    "compute_on_cpu": True
}

SETTING_3 = {
    "thresholds": NUM_THRESHOLD,
    "num_labels": NUM_LABELS,
    "average": AVERAGE,
    "compute_on_cpu": True
}

METRIC_DICT = {
    "iou": MultilabelJaccardIndex(**SETTING_1),
    "accuracy": MultilabelAccuracy(**SETTING_1),
    "precision": MultilabelPrecision(**SETTING_1),
    "f1": MultilabelF1Score(**SETTING_1),
    "recall": MultilabelRecall(**SETTING_1),
    "stat": MultilabelStatScores(**SETTING_1),
    "roc": MultilabelROC(**SETTING_2),
    "auroc": MultilabelAUROC(**SETTING_3),
    "conf": MultilabelConfusionMatrix(**SETTING_2)
}


class Evaluator(object):

    def __init__(
            self,
            model,
            dataloader,
            device,
            report_metrics=True,
            metrics=["iou"],
            num_time_step_previous=10,
            num_time_step_predict=10,
            threshold=0.5,
            vehicle_threshold=0.3,
            bev_selected_channels=None,
            save_path=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.report_metrics = report_metrics
        self.metrics = metrics
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_predict = num_time_step_predict
        self.threshold = threshold
        self.vehicle_threshold = vehicle_threshold
        self.save_path = save_path
        self.bev_selected_channels = bev_selected_channels
        # Create folder at save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.model.to(self.device)

    def evaluate(self, render=True, save=True):

        self.model.eval()

        with torch.no_grad():
            for i, (data) in enumerate(self.dataloader):

                world_future_bev_predicted_list = []

                world_previous_bev = data["bev_world"]["bev"][:, :self.num_time_step_previous].to(
                    self.device).clone()
                world_future_bev = data["bev_world"]["bev"][:, self.num_time_step_previous:
                                                            self.num_time_step_previous + self.num_time_step_predict].to(self.device).clone()

                for k in range(self.num_time_step_predict):

                    # Predict the future bev
                    world_future_bev_predicted = self.model(
                        world_previous_bev, sample_latent=True)

                    #  Append the predicted future bev to the list
                    world_future_bev_predicted_list.append(
                        world_future_bev_predicted)

                    # Update the previous bev
                    world_previous_bev = torch.cat((world_previous_bev[:, 1:], torch.sigmoid(
                        world_future_bev_predicted.unsqueeze(1))), dim=1)

                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1)

                if self.report_metrics:

                    for metric in self.metrics:

                        metric_ = METRIC_DICT[metric].to(self.device)

                        world_future_bev_predicted_ = world_future_bev_predicted.permute(
                            0, 2, 1, 3, 4).clone()
                        world_future_bev_ = world_future_bev.permute(
                            0, 2, 1, 3, 4).clone().to(torch.uint8)

                        metric_.update(
                            world_future_bev_predicted_,
                            world_future_bev_)

                        # if isinstance(result, torch.Tensor):
                        #     metric_dict[metric] = result.cpu().numpy()
                        # elif isinstance(result, tuple):
                        #     metric_dict[metric] = [res.cpu().numpy()
                        #                            for res in result]
                        # else:
                        #     raise ValueError(
                        #         "Metric result must be a torch.Tensor or a tuple of torch.Tensor")

                    # metric_dict_list.append(metric_dict)

                self._init_canvas(world_previous_bev.shape[0])
                self._draw(data, world_future_bev_predicted_list,
                           None)

                if render:

                    canvas_scaled_half = cv2.resize(
                        self.canvas, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("Evaluation", canvas_scaled_half)

                if save:

                    self._save(i)

        if self.report_metrics:

            metric_table = {}
            metric_plot = {}
            for metric in self.metrics:

                metric_ = METRIC_DICT[metric]
                result = metric_.compute()

                if isinstance(result, torch.Tensor):
                    result = result.cpu().numpy()
                elif isinstance(result, tuple):
                    result = [res.cpu().numpy()
                              for res in result]
                else:
                    raise ValueError(
                        "Metric result must be a torch.Tensor or a tuple of torch.Tensor")

                if metric in ["roc", "auroc", "stat"]:

                    metric_plot[metric] = result

                else:

                    metric_table[metric] = result

            # Write to a csv file
            df = pd.DataFrame(metric_table)
            df.to_csv(os.path.join(self.save_path, "metrics.csv"))

            # Plot the roc curve
            if "roc" in self.metrics and "auroc" in self.metrics:
                fpr, tpr, thresholds = metric_plot["roc"]
                auroc = metric_plot["auroc"]
                plot_roc(
                    fpr,
                    tpr,
                    thresholds,
                    auroc,
                    self.save_path,
                    multi=True)

            if "stat" in self.metrics:
                tp = metric_plot["stat"][..., 0]
                fp = metric_plot["stat"][..., 1]
                tn = metric_plot["stat"][..., 2]
                fn = metric_plot["stat"][..., 3]

                plot_confusion_matrix(
                    tp, fp, tn, fn, self.save_path, multi=True)

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

                world_future_bev_predicted = torch.sigmoid(
                    world_future_bev_predicted_list[j][canvas_num])

                world_future_bev_predicted_vehicle = world_future_bev_predicted[:, :, -2]
                world_future_bev_predicted_vehicle = (
                    world_future_bev_predicted_vehicle > self.vehicle_threshold).float()
                world_future_bev_predicted = (
                    world_future_bev_predicted > self.threshold).float()
                world_future_bev_predicted[:, :, -
                                           2] = world_future_bev_predicted_vehicle

                self.canvas[:data["bev_world"]["bev"].shape[-2], (j + self.num_time_step_previous) * data["bev_world"]["bev"].shape[-1]:(j + self.num_time_step_previous + 1)
                            * data["bev_world"]["bev"].shape[-1]] = cv2.cvtColor(self._bev_to_rgb(world_future_bev_predicted.clone().detach().cpu().numpy()), cv2.COLOR_RGB2BGR)
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

        # rgb_image = BirdViewProducer.as_rgb_with_indices(
        #     bev, [0, 5, 6, 8, 9, 9, 10, 11])
        # return rgb_image

        # New BEV
        if self.bev_selected_channels is None:
            rgb_image = BirdViewProducer.as_rgb(bev)
        else:
            rgb_image = BirdViewProducer.as_rgb_with_indices(
                bev, self.bev_selected_channels)
            # rgb_image = BirdViewProducer.as_rgb_with_indices(
            #     bev[...,6:7], [6])
        return rgb_image

    def _save(self, step):

        if self.save_path is not None:

            for k in range(len(self.canvas_list)):
                cv2.imwrite(
                    f"{self.save_path}/{k + (step * len(self.canvas_list))}.png",
                    self.canvas_list[k])
