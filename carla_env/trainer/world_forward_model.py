import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from carla_env.sampler.distributed_weighted_sampler import DistributedWeightedSampler

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
        self,
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        rank,
        lr_scheduler=None,
        reconstruction_loss=torch.nn.MSELoss(),
        sigmoid_before_loss=False,
        save_interval=6,
        val_interval=3,
        num_time_step_previous=10,
        num_time_step_future=10,
        num_epochs=1000,
        current_epoch=0,
        logvar_clip=True,
        logvar_clip_min=-5,
        logvar_clip_max=5,
        gradient_clip=True,
        gradient_clip_type="norm",
        gradient_clip_value=0.3,
        save_path=None,
        train_step=0,
        val_step=0,
    ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.rank = rank
        self.lr_scheduler = lr_scheduler
        self.reconstruction_loss = reconstruction_loss
        self.sigmoid_before_loss = sigmoid_before_loss
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.logvar_clip = logvar_clip
        self.logvar_clip_min = logvar_clip_min
        self.logvar_clip_max = logvar_clip_max
        self.gradient_clip = gradient_clip
        self.gradient_clip_type = gradient_clip_type
        self.gradient_clip_value = gradient_clip_value
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step

        # ----------------------- Save same values know a prior ---------------------- #
        (self.B_TRAIN, _, self.C, self.H, self.W) = next(iter(dataloader_train))[
            "bev_world"
        ]["bev"].shape

        (self.B_VAL, _, _, _, _) = next(iter(dataloader_val))["bev_world"]["bev"].shape

        logger.info(f"Batch size train: {self.B_TRAIN}")
        logger.info(f"Batch size val: {self.B_VAL}")
        logger.info(f"Number of channels: {self.C}")
        logger.info(f"Height: {self.H}")
        logger.info(f"Width: {self.W}")

        # ------------------------------------ ... ----------------------------------- #

        # ---------------------------------------------------------------------------- #
        #                                   DDP SETUP                                  #
        # ---------------------------------------------------------------------------- #
        self.model.to(self.rank)
        self.model = DDP(self.model, device_ids=[self.rank])

    def train(self, epoch, run):

        self.model.train()

        if isinstance(
            self.dataloader_train.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_train.sampler.set_epoch(epoch)

        logger.info("Training routine started!")

        for i, (data) in enumerate(self.dataloader_train):

            loss_dict = self.shared_step(i, data)
            loss = loss_dict["loss"]
            loss_kl_div = loss_dict["loss_kl_div"]
            loss_reconstruction = loss_dict["loss_reconstruction"]
            self.optimizer.zero_grad()
            loss.backward()

            # Clip the gradients
            if self.gradient_clip:
                if self.gradient_clip_type == "norm":
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_value
                    )
                else:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.gradient_clip_value
                    )

            # Update the weights
            self.optimizer.step()

            self.train_step += self.B_TRAIN

            if run is not None:
                run.log(
                    {
                        "train/step": self.train_step,
                        "train/loss": loss,
                        "train/loss_kl_divergence": loss_kl_div,
                        "train/loss_reconstruction": loss_reconstruction,
                    },
                    commit=True,
                )
        logger.info("Training routine finished!")

    def validate(self, epoch, run=None):

        self.model.eval()

        if isinstance(
            self.dataloader_val.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_val.sampler.set_epoch(epoch)

        logger.info("Validation routine started!")
        losses_total = []
        losses_kl_div = []
        losses_reconstruction = []

        with torch.no_grad():

            for i, (data) in enumerate(self.dataloader_val):

                loss_dict = self.shared_step(i, data)

                loss = loss_dict["loss"]
                loss_kl_div = loss_dict["loss_kl_div"]
                loss_reconstruction = loss_dict["loss_reconstruction"]

                losses_total.append(loss.item())
                losses_kl_div.append(loss_kl_div.item())
                losses_reconstruction.append(loss_reconstruction.item())

                self.val_step += self.B_VAL

        loss = np.mean(losses_total)
        loss_kl_div = np.mean(losses_kl_div)
        loss_reconstruction = np.mean(losses_reconstruction)

        if run is not None:

            if self.lr_scheduler is not None:
                run.log({"val/lr": self.lr_scheduler.get_last_lr()[0]}, commit=False)

            run.log(
                {
                    "val/step": self.val_step,
                    "val/loss": loss,
                    "val/loss_kl_divergence": loss_kl_div,
                    "val/loss_reconstruction": loss_reconstruction,
                },
                commit=True,
            )

        logger.info("Validation routine finished!")
        return {
            "loss": loss,
            "loss_kl_div": loss_kl_div,
            "loss_reconstruction": loss_reconstruction,
        }

    def shared_step(self, i, data):

        world_previous_bev = data["bev_world"]["bev"][
            :, : self.num_time_step_previous
        ].to(self.rank)

        world_future_bev = data["bev_world"]["bev"][
            :,
            self.num_time_step_previous : self.num_time_step_previous
            + self.num_time_step_future,
        ].to(self.rank)

        world_future_bev_predicted_list = []
        mu_list = []
        logvar_list = []

        for k in range(self.num_time_step_future):

            # Predict the future bev
            world_future_bev_predicted, mu, logvar = self.model(
                world_previous_bev, world_future_bev[:, k]
            )

            world_future_bev_predicted = world_future_bev_predicted.clamp(-10, 10)

            # Append the predicted bev
            world_future_bev_predicted_list.append(world_future_bev_predicted)
            mu_list.append(mu)
            logvar_list.append(logvar)

            # Update the previous bev
            world_previous_bev = torch.cat(
                (
                    world_previous_bev[:, 1:],
                    torch.sigmoid(world_future_bev_predicted).unsqueeze(1),
                ),
                dim=1,
            )

        # Stack the predicted bev
        world_future_bev_predicted = torch.stack(world_future_bev_predicted_list, dim=1)

        # Stack the mu and logvar
        mu = torch.stack(mu_list, dim=1)
        logvar = torch.stack(logvar_list, dim=1)

        # Clip the logvar
        if self.logvar_clip:
            logvar = torch.clamp(logvar, self.logvar_clip_min, self.logvar_clip_max)

        # Compute the reconstruction loss
        if self.sigmoid_before_loss:

            world_future_bev_predicted = torch.sigmoid(world_future_bev_predicted)

        if isinstance(
            self.reconstruction_loss,
            (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss),
        ):

            world_future_bev_predicted = world_future_bev_predicted.permute(
                0, 1, 3, 4, 2
            )
            world_future_bev = world_future_bev.permute(0, 1, 3, 4, 2)

        loss_reconstruction = self.reconstruction_loss(
            input=world_future_bev_predicted, target=world_future_bev
        )

        if self.logvar_clip:
            logvar = torch.clamp(logvar, self.logvar_clip_min, self.logvar_clip_max)

        # Compute the KL divergence
        loss_kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Compute the total loss
        loss = loss_reconstruction + loss_kl_div

        return {
            "loss": loss,
            "loss_kl_div": loss_kl_div,
            "loss_reconstruction": loss_reconstruction,
        }

    def learn(self, run=None):

        # loss_dict = self.validate(0, run)
        # logger.info(f"{'*' * 10} Initial Validation {'*' * 10}")
        # logger.info(f"Epoch: Start")
        # for key, value in loss_dict.items():
        #     logger.info(f"{key}: {value}")
        # logger.info(f"{'*' * 10} Initial Validation {'*' * 10}")

        for epoch in range(self.current_epoch, self.num_epochs):

            self.train(epoch, run)

            if (epoch + 1) % self.val_interval == 0:

                loss_dict = self.validate(epoch, run)
                logger.info(f"{'*' * 10} Validation {'*' * 10}")
                logger.info(f"Epoch: {epoch + 1}")
                for key, value in loss_dict.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"{'*' * 10} Validation {'*' * 10}")

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if (
                ((epoch + 1) % self.save_interval == 0)
                and (self.save_path is not None)
                and (self.rank == 0)
            ):

                torch.save(
                    {
                        "model_state_dict": self.model.module.state_dict(),
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

                logger.info(
                    f"Saving model to {self.save_path / Path(f'checkpoint_{epoch}.pt')}"
                )

                if run is not None:
                    run.save(str(self.save_path / Path(f"checkpoint_{epoch}.pt")))
