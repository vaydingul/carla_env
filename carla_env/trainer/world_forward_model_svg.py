import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from tqdm import tqdm

from carla_env.sampler.distributed_weighted_sampler import DistributedWeightedSampler

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def kl_divergence(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(sigma2_2) / sqrt(sigma2_1) ) +
    #   (sigma2_1 + (mu_1 - mu_2)^2) / (2 * sigma2_2) - 1/2

    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = (
        torch.log(sigma2 / sigma1)
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2))
        - 1 / 2
    )
    return kld.sum()


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
        kl_divergence_beta=1e-4,
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
        self.kl_divergence_beta = kl_divergence_beta
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step

        # ---------------------------------------------------------------------------- #
        #                                   DDP SETUP                                  #
        # ---------------------------------------------------------------------------- #
        self.model.to(self.rank)
        self.model = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False)

    def train(self, epoch, run):
        self.model.train()

        if isinstance(
            self.dataloader_train.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_train.sampler.set_epoch(epoch)

        logger.info(f"Training {epoch} started!")

        # Iterate over dataloader with a fancy progress bar

        for i, data in tqdm(
            enumerate(self.dataloader_train),
            total=len(self.dataloader_train),
            colour="red",
        ):
            (loss_dict, B) = self.shared_step(i, data)
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

            self.train_step += B

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
        logger.info(f"Training {epoch} finished!")

    def validate(self, epoch, run=None):
        self.model.eval()

        if isinstance(
            self.dataloader_val.sampler,
            (DistributedWeightedSampler, DistributedSampler),
        ):
            self.dataloader_val.sampler.set_epoch(epoch)

        logger.info(f"Validation {epoch} started!")
        losses_total = []
        losses_kl_div = []
        losses_reconstruction = []

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(self.dataloader_val),
                total=len(self.dataloader_val),
                colour="green",
            ):
                (loss_dict, B) = self.shared_step(i, data)

                loss = loss_dict["loss"]
                loss_kl_div = loss_dict["loss_kl_div"]
                loss_reconstruction = loss_dict["loss_reconstruction"]

                losses_total.append(loss.item())
                losses_kl_div.append(loss_kl_div.item())
                losses_reconstruction.append(loss_reconstruction.item())

                self.val_step += B

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

        logger.info(f"Validation {epoch} finished!")
        return {
            "loss": loss,
            "loss_kl_div": loss_kl_div,
            "loss_reconstruction": loss_reconstruction,
        }

    def shared_step(self, i, data):
        world_bev = data["bev_world"]["bev"].to(self.rank)
        # world_future_bev = world_bev[:, 1:].clone()
        B, S, _, _, _ = world_bev.shape

        world_future_bev_predicted_list = []
        mu_posterior_list = []
        logvar_posterior_list = []
        mu_prior_list = []
        logvar_prior_list = []

        self.model.module.init_hidden(B, device=self.rank)

        for k in range(1, self.num_time_step_previous + self.num_time_step_future):
            # Predict the future bev

            if k < self.num_time_step_previous:
                skip_feature = None

            elif k == self.num_time_step_previous:
                skip_feature = output["skip_feature"]

            else:
                skip_feature = skip_feature

            output = self.model(
                world_bev[:, k - 1],
                world_bev[:, k],
                skip_feature=skip_feature,
            )

            # skip_feature = output["skip_feature"]

            # world_future_bev_predicted = world_future_bev_predicted.clamp(-10, 10)

            # Append the predicted bev
            world_future_bev_predicted_list.append(output["world_future_bev_predicted"])
            mu_posterior_list.append(output["mu_posterior"])
            logvar_posterior_list.append(output["logvar_posterior"])
            mu_prior_list.append(output["mu_prior"])
            logvar_prior_list.append(output["logvar_prior"])

        # Stack everything
        world_future_bev_predicted = torch.stack(world_future_bev_predicted_list, dim=1)
        mu_posterior = torch.stack(mu_posterior_list, dim=1)
        logvar_posterior = torch.stack(logvar_posterior_list, dim=1)
        mu_prior = torch.stack(mu_prior_list, dim=1)
        logvar_prior = torch.stack(logvar_prior_list, dim=1)

        # Clip the logvar
        if self.logvar_clip:
            logvar_prior = torch.clamp(
                logvar_prior, self.logvar_clip_min, self.logvar_clip_max
            )
            logvar_posterior = torch.clamp(
                logvar_posterior, self.logvar_clip_min, self.logvar_clip_max
            )

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
            world_future_bev = world_bev[:, 1:].permute(0, 1, 3, 4, 2)

        loss_reconstruction = self.reconstruction_loss(
            input=world_future_bev_predicted, target=world_future_bev
        )

        # Compute the KL divergence
        loss_kl_div = kl_divergence(
            mu_posterior, logvar_posterior, mu_prior, logvar_prior
        ) / (B * (S - 1))

        # Compute the total loss
        loss = loss_reconstruction + (loss_kl_div * self.kl_divergence_beta)

        return {
            "loss": loss,
            "loss_kl_div": loss_kl_div,
            "loss_reconstruction": loss_reconstruction,
        }, B

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

                run.save(str(self.save_path / Path(f"checkpoint_{epoch}.pt")))
