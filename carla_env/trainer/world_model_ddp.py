import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import (
    MultilabelJaccardIndex,
    MultilabelPrecision,
    MultilabelRecall,
)
from pathlib import Path
logger = logging.getLogger(__name__)


NUM_LABELS = 8
THRESHOLD = 0.5
NUM_THRESHOLD = 10
AVERAGE = None

SETTING_1 = {
    "threshold": THRESHOLD,
    "average": AVERAGE,
    "num_labels": NUM_LABELS,
}


METRIC_DICT = {
    "iou": MultilabelJaccardIndex(**SETTING_1),
    "precision": MultilabelPrecision(**SETTING_1),
    "recall": MultilabelRecall(**SETTING_1),
}


class Trainer(object):

    def __init__(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            gpu_id,
            save_every=6,
            val_every=3,
            num_time_step_previous=10,
            num_time_step_future=10,
            num_epochs=1000,
            report_metrics=True,
            metrics=["iou, precision, recall"],
            current_epoch=0,
            reconstruction_loss="mse_loss",
            bev_channel_weights=None,
            logvar_clip=True,
            logvar_clip_min=-5,
            logvar_clip_max=5,
            lr_scheduler=None,
            gradient_clip_type="norm",
            gradient_clip_value=0.3,
            save_path=None,
            train_step=0,
            val_step=0):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.val_every = val_every
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.num_epochs = num_epochs
        self.report_metrics = report_metrics
        self.metrics = metrics
        self.current_epoch = current_epoch
        self.reconstruction_loss = F.mse_loss if reconstruction_loss == "mse_loss" else F.binary_cross_entropy_with_logits
        self.bev_channel_weights = bev_channel_weights
        self.logvar_clip = logvar_clip
        self.logvar_clip_min = logvar_clip_min
        self.logvar_clip_max = logvar_clip_max
        self.lr_scheduler = lr_scheduler
        self.gradient_clip_type = gradient_clip_type
        self.gradient_clip_value = gradient_clip_value
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step

        self.b, _, self.c, self.h, self.w = next(
            iter(dataloader_train))["bev_world"]["bev"].shape

        if self.bev_channel_weights is not None:
            self.weight = torch.Tensor(
                self.bev_channel_weights).to(
                self.gpu_id)

        self.metrics_ = [METRIC_DICT[metric].to(
            self.gpu_id) for metric in self.metrics]

        self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def train(self, run):

        self.model.train()

        for i, (data) in enumerate(self.dataloader_train):

            world_previous_bev = data["bev_world"]["bev"][:,
                                                          :self.num_time_step_previous].to(self.gpu_id)
            world_future_bev = data["bev_world"]["bev"][:, self.num_time_step_previous:
                                                        self.num_time_step_previous + self.num_time_step_future].to(self.gpu_id)

            world_future_bev_predicted_list = []
            mu_list = []
            logvar_list = []

            for k in range(self.num_time_step_future):

                # Predict the future bev
                world_future_bev_predicted, mu, logvar = self.model(
                    world_previous_bev, world_future_bev[:, k])

                world_future_bev_predicted = world_future_bev_predicted.clamp(
                    -10, 10)

                # Append the predicted bev
                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)
                mu_list.append(mu)
                logvar_list.append(logvar)

                # Update the previous bev
                world_previous_bev = torch.cat((world_previous_bev[:, 1:], torch.sigmoid(
                    world_future_bev_predicted).unsqueeze(1)), dim=1)

            # Stack the predicted bev
            world_future_bev_predicted = torch.stack(
                world_future_bev_predicted_list, dim=1)

            # Stack the mu and logvar
            mu = torch.stack(mu_list, dim=1)
            logvar = torch.stack(logvar_list, dim=1)

            # Clip the logvar
            if self.logvar_clip:
                logvar = torch.clamp(logvar, self.logvar_clip_min,
                                     self.logvar_clip_max)

            # Compute the reconstruction loss
            if self.reconstruction_loss == F.mse_loss:
                world_future_bev_predicted = torch.sigmoid(
                    world_future_bev_predicted)
                loss_reconstruction = self.reconstruction_loss(
                    input=world_future_bev_predicted, target=world_future_bev)
            else:
                if self.bev_channel_weights is not None:
                    loss_reconstruction = self.reconstruction_loss(
                        input=world_future_bev_predicted.permute(
                            0, 1, 3, 4, 2), target=world_future_bev.permute(
                            0, 1, 3, 4, 2), pos_weight=self.weight)
                else:

                    loss_reconstruction = self.reconstruction_loss(
                        input=world_future_bev_predicted, target=world_future_bev)

            # Compute the KL divergence
            loss_kl_div = -0.5 * \
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Compute the total loss
            loss = loss_reconstruction + loss_kl_div

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
                         "train/loss": loss,
                         "train/loss_kl_divergence": loss_kl_div,
                         "train/loss_reconstruction": loss_reconstruction})

    def validate(self, run=None):

        self.model.eval()

        losses_total = []
        losses_kl_div = []
        losses_reconstruction = []

        with torch.no_grad():

            for i, (data) in enumerate(self.dataloader_val):

                world_previous_bev = data["bev_world"]["bev"][:,
                                                              :self.num_time_step_previous].to(self.gpu_id)
                world_future_bev = data["bev_world"]["bev"][:, self.num_time_step_previous:
                                                            self.num_time_step_previous + self.num_time_step_future].to(self.gpu_id)

                world_future_bev_predicted_list = []
                mu_list = []
                logvar_list = []

                for k in range(self.num_time_step_future):

                    # Predict the future bev
                    world_future_bev_predicted, mu, logvar = self.model(
                        world_previous_bev, world_future_bev[:, k])

                    world_future_bev_predicted = world_future_bev_predicted.clamp(
                        -10, 10)

                    world_future_bev_predicted_list.append(
                        world_future_bev_predicted)
                    mu_list.append(mu)
                    logvar_list.append(logvar)

                    # Update the previous bev
                    world_previous_bev = torch.cat((world_previous_bev[:, 1:], torch.sigmoid(
                        world_future_bev_predicted).unsqueeze(1)), dim=1)

                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1)
                mu = torch.stack(mu_list, dim=1)
                logvar = torch.stack(logvar_list, dim=1)

                if self.logvar_clip:
                    logvar = torch.clamp(logvar, self.logvar_clip_min,
                                         self.logvar_clip_max)

                # Compute the reconstruction loss
                if self.reconstruction_loss == F.mse_loss:
                    world_future_bev_predicted = torch.sigmoid(
                        world_future_bev_predicted)
                    loss_reconstruction = self.reconstruction_loss(
                        input=world_future_bev_predicted, target=world_future_bev)
                else:
                    if self.bev_channel_weights is not None:
                        loss_reconstruction = self.reconstruction_loss(
                            input=world_future_bev_predicted.permute(
                                0, 1, 3, 4, 2), target=world_future_bev.permute(
                                0, 1, 3, 4, 2), pos_weight=self.weight)
                    else:

                        loss_reconstruction = self.reconstruction_loss(
                            input=world_future_bev_predicted, target=world_future_bev)
                if self.report_metrics:
                    for metric_ in self.metrics_:
                        world_future_bev_predicted_ = world_future_bev_predicted.permute(
                            0, 2, 1, 3, 4).clone()
                        world_future_bev_ = world_future_bev.permute(
                            0, 2, 1, 3, 4).clone().to(torch.uint8)

                        metric_.update(world_future_bev_predicted_,
                                       world_future_bev_)

                # Calculate the KL divergence loss
                loss_kl_div = -0.5 * \
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = loss_kl_div + loss_reconstruction

                losses_total.append(loss.item())
                losses_kl_div.append(loss_kl_div.item())
                losses_reconstruction.append(loss_reconstruction.item())

                self.val_step += world_previous_bev.shape[0]

        loss = np.mean(losses_total)
        loss_kl_div = np.mean(losses_kl_div)
        loss_reconstruction = np.mean(losses_reconstruction)

        if run is not None:
            run.log({"val/step": self.val_step,
                     "val/loss": loss,
                     "val/loss_kl_divergence": loss_kl_div,
                     "val/loss_reconstruction": loss_reconstruction})
            if self.lr_scheduler is not None:
                run.log({"val/lr": self.lr_scheduler.get_last_lr()[0]})
            if self.report_metrics:
                for (metric, metric_) in zip(self.metrics, self.metrics_):
                    result = float(metric_.compute().cpu().numpy())
                    run.log({"eval/{}".format(metric): result})

        return loss, loss_kl_div, loss_reconstruction

    def learn(self, run=None):

        for epoch in range(self.current_epoch, self.num_epochs):

            loss, loss_kl_div, loss_reconstruction = self.validate(run)

            self.train(run)

            if (epoch + 1) % self.val_every == 0:

                loss, loss_kl_div, loss_reconstruction = self.validate(run)
                logger.info(
                    "Epoch: {}, Val Loss: {}, Val Loss KL Div: {}, Val Loss Reconstruction: {}".format(
                        epoch, loss, loss_kl_div, loss_reconstruction))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if ((epoch + 1) % self.save_every ==
                    0) and self.save_path is not None:

                torch.save({
                    "model_state_dict": self.model.module.state_dict(),
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
