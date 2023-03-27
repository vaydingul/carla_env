import torch
import cv2
import numpy as np
from carla_env.bev import BirdViewProducer, BirdViewMasks
from carla_env.renderer.renderer import Renderer
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
    MultilabelStatScores,
)
import pandas as pd
from utils.plot_utils import plot_roc, plot_confusion_matrix
import wandb

NUM_LABELS = 7
THRESHOLD = 0.5
NUM_THRESHOLD = 10
AVERAGE = None

SETTING_1 = {
    "threshold": THRESHOLD,
    "average": AVERAGE,
    "num_labels": NUM_LABELS,
    "compute_on_cpu": True,
}

SETTING_2 = {
    "thresholds": NUM_THRESHOLD,
    "num_labels": NUM_LABELS,
    "compute_on_cpu": True,
}

SETTING_3 = {
    "thresholds": NUM_THRESHOLD,
    "num_labels": NUM_LABELS,
    "average": AVERAGE,
    "compute_on_cpu": True,
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
    "conf": MultilabelConfusionMatrix(**SETTING_2),
}


class Evaluator(object):
    def __init__(
        self,
        model,
        dataset,
        dataloader,
        device,
        renderer=None,
        metrics=["iou"],
        num_time_step_previous=10,
        num_time_step_predict=10,
        thresholds=0.5,
        wandb_log_interval=10,
        save_path=None,
    ):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.metrics = metrics
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_predict = num_time_step_predict
        self.thresholds = thresholds
        self.wandb_log_interval = wandb_log_interval
        self.save_path = save_path

        self.bev_selected_channels = self.dataset.bev_selected_channels

        if isinstance(self.thresholds, float):
            self.thresholds = [self.thresholds] * len(self.bev_selected_channels)

        (self.B, self.S, self.C, self.H, self.W) = next(iter(dataloader))["bev_world"][
            "bev"
        ].shape

        if renderer is not None:

            renderer["width"] = (self.W + 10) * self.S
            renderer["height"] = (self.H + 20) * 2
            renderer["save_path"] = self.save_path
            
            renderer = Renderer(config=renderer)
            
            self.save_path = renderer.save_path

            self.renderer_list = [renderer] * self.B

        self.model.to(self.device)

    def evaluate(self, run=None):

        self.model.eval()

        with torch.no_grad():

            for i, (data) in enumerate(self.dataloader):

                world_future_bev_predicted_list = []

                world_previous_bev = (
                    data["bev_world"]["bev"][:, : self.num_time_step_previous]
                    .to(self.device)
                    .clone()
                )
                world_previous_bev_render = world_previous_bev.clone()
                world_future_bev = (
                    data["bev_world"]["bev"][
                        :,
                        self.num_time_step_previous : self.num_time_step_previous
                        + self.num_time_step_predict,
                    ]
                    .to(self.device)
                    .clone()
                )
                world_future_bev_render = world_future_bev.clone()

                for k in range(self.num_time_step_predict):

                    # Predict the future bev
                    (_, world_future_bev_predicted) = self.model(
                        world_previous_bev, sample_latent=True
                    )

                    #  Append the predicted future bev to the list
                    world_future_bev_predicted_list.append(world_future_bev_predicted)

                    # Update the previous bev
                    world_previous_bev = torch.cat(
                        (
                            world_previous_bev[:, 1:],
                            torch.sigmoid(world_future_bev_predicted.unsqueeze(1)),
                        ),
                        dim=1,
                    )

                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1
                )
                world_future_bev_predicted_render = world_future_bev_predicted.clone()

                for metric in self.metrics:

                    metric_ = METRIC_DICT[metric].to(self.device)

                    world_future_bev_predicted_ = world_future_bev_predicted.permute(
                        0, 2, 1, 3, 4
                    ).clone()
                    world_future_bev_ = (
                        world_future_bev.permute(0, 2, 1, 3, 4).clone().to(torch.uint8)
                    )

                    metric_.update(world_future_bev_predicted_, world_future_bev_)

                image_paths = self.render(
                    i,
                    world_previous_bev_render,
                    world_future_bev_render,
                    world_future_bev_predicted_render,
                )

        metric_table = {}
        metric_plot = {}
        for metric in self.metrics:

            metric_ = METRIC_DICT[metric]
            result = metric_.compute()

            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            elif isinstance(result, tuple):
                result = [res.cpu().numpy() for res in result]
            else:
                raise ValueError(
                    "Metric result must be a torch.Tensor or a tuple of torch.Tensor"
                )

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
            plot_roc(fpr, tpr, thresholds, auroc, self.save_path, multi=True)

        if "stat" in self.metrics:
            tp = metric_plot["stat"][..., 0]
            fp = metric_plot["stat"][..., 1]
            tn = metric_plot["stat"][..., 2]
            fn = metric_plot["stat"][..., 3]

            plot_confusion_matrix(tp, fp, tn, fn, self.save_path, multi=True)

        if run is not None:
            run.log({"Metrics": wandb.Table(dataframe=df)})

            for k in range(0, len(image_paths), self.wandb_log_interval):
                run.log({"Images": wandb.Image(image_paths[k])})

    def render(
        self,
        i,
        world_previous_bev,
        world_future_bev,
        world_future_bev_predicted,
    ):

        saved_image_paths = []

        for (renderer_num, renderer) in enumerate(self.renderer_list):

            renderer.reset()

            # Draw the previous bev
            for j in range(self.num_time_step_previous):

                bev_ = (
                    world_previous_bev[renderer_num, j].clone().detach().cpu().numpy()
                )

                renderer.render_image(self._bev_to_rgb(bev_), move_cursor="right")

                cursor_ = renderer.get_cursor()

                renderer.move_cursor(direction="left-down", amount=(20, self.W // 2))

                renderer.render_text(
                    f"GT t - {self.num_time_step_previous - j -1}",
                    move_cursor="right",
                    font_thickness=2,
                )

                cursor_ = (cursor_[0], cursor_[1] + 10)

                renderer.move_cursor(direction="point", amount=cursor_)

            cursor_future = renderer.get_cursor()

            # Draw the predicted future bev
            for j in range(self.num_time_step_predict):

                bev_ = (
                    torch.sigmoid(world_future_bev_predicted)[renderer_num, j]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                )

                for k in range(self.C):

                    world_future_bev_predicted_ = bev_[k]
                    world_future_bev_predicted_[world_future_bev_predicted_ > 0.5] = 1
                    world_future_bev_predicted_[world_future_bev_predicted_ <= 0.5] = 0
                    bev_[k] = world_future_bev_predicted_

                renderer.render_image(self._bev_to_rgb(bev_), move_cursor="right")

                cursor_ = renderer.get_cursor()

                renderer.move_cursor(direction="left-down", amount=(20, self.W // 2))

                renderer.render_text(
                    f"Pred t + {j + 1}", move_cursor="right", font_thickness=2
                )

                cursor_ = (cursor_[0], cursor_[1] + 10)

                renderer.move_cursor(direction="point", amount=cursor_)

            cursor_ = (cursor_future[0] + 10 + self.H, cursor_future[1])

            renderer.move_cursor(direction="point", amount=cursor_)

            # Draw the ground truth future bev below line
            for j in range(self.num_time_step_predict):

                bev_ = world_future_bev[renderer_num, j].clone().detach().cpu().numpy()

                renderer.render_image(self._bev_to_rgb(bev_), move_cursor="right")

                cursor_ = renderer.get_cursor()

                renderer.move_cursor(direction="left-down", amount=(20, self.W // 2))

                renderer.render_text(
                    f"GT t + {j + 1}", move_cursor="right", font_thickness=2
                )

                cursor_ = (cursor_[0], cursor_[1] + 10)

                renderer.move_cursor(direction="point", amount=cursor_)

            renderer.show()
            saved_image_path = renderer.save(info=f"{renderer_num + (self.B * i)}")
            saved_image_paths.append(saved_image_path)

        return saved_image_paths

    def _bev_to_rgb(self, bev):

        # Transpose the bev representation
        bev = bev.transpose(1, 2, 0)

        # New BEV

        rgb_image = BirdViewProducer.as_rgb_with_indices(
            bev, self.bev_selected_channels
        )

        rgb_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        # Old BEV

        # rgb_image = BirdViewProducer.as_rgb_with_indices(
        #     bev, [0, 5, 6, 8, 9, 9, 10, 11])
        # return rgb_image

        return rgb_image
