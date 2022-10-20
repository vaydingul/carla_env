import logging
import numpy as np
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Trainer(object):
    # TODO: Change according to the new dataloader

    def __init__(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            device,
            num_epochs=1000,
            log_interval=10):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.train_step = 0
        self.val_step = 0

        self.model.to(self.device)

    def train(self, run):

        self.model.train()

        for i, (data) in enumerate(self.dataloader_train):

            world_current_bev = data["bev"][:, :-1].to(self.device)
            world_future_bev = data["bev"][:, -2:-1].to(self.device)

            # Predict the future bev
            world_future_bev_predicted, mu, logvar = self.model(
                world_current_bev, world_future_bev)

            # Calculate the KL divergence loss
            loss_kl_div = -0.5 * \
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Calculate the reconstruction loss
            loss_reconstruction = F.mse_loss(
                world_future_bev_predicted, world_future_bev.squeeze())

            loss = loss_kl_div + loss_reconstruction

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_step += world_current_bev.shape[0]

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

                world_current_bev = data["bev"][:, :-1].to(self.device)
                world_future_bev = data["bev"][:, -2:-1].to(self.device)
                # Predict the future bev
                world_future_bev_predicted, mu, logvar = self.model(
                    world_current_bev, world_future_bev)

                # Calculate the KL divergence loss
                loss_kl_div = -0.5 * \
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Calculate the reconstruction loss
                loss_reconstruction = F.mse_loss(
                    world_future_bev_predicted, world_future_bev)

                loss = loss_kl_div + loss_reconstruction

                losses_total.append(loss.item())
                losses_kl_div.append(loss_kl_div.item())
                losses_reconstruction.append(loss_reconstruction.item())

                self.val_step += world_current_bev.shape[0]

        loss = np.mean(losses_total)
        loss_kl_div = np.mean(losses_kl_div)
        loss_reconstruction = np.mean(losses_reconstruction)

        if run is not None:
            run.log({"val/step": self.val_step,
                     "val/loss": loss,
                     "val/loss_location": loss_kl_div,
                     "val/loss_rotation": loss_reconstruction})

        return loss, loss_kl_div, loss_reconstruction

    def learn(self, run=None):

        for epoch in range(self.num_epochs):

            self.train(run)
            loss, loss_kl_div, loss_reconstruction = self.validate(run)
            logger.info(
                "Epoch: {}, Val Loss: {}, Val Loss KL Div: {}, Val Loss Reconstruction: {}".format(
                    epoch, loss, loss_kl_div, loss_reconstruction))
