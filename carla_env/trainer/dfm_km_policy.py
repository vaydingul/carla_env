import logging
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            device,
            cost,
            num_time_step_previous=10,
            num_time_step_future=10,
            num_epochs=1000,
            current_epoch=0,
            lr_scheduler=None,
            save_path=None,
            train_step=0,
            val_step=0):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.device = device
        self.cost = cost
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.lr_scheduler = lr_scheduler
        self.save_path = save_path
        self.train_step = train_step
        self.val_step = val_step

        self.model.to(self.device)

    def train(self, run):

        self.model.train()

        for i, (data) in enumerate(self.dataloader_train):

            world_previous_bev = data["bev"]["bev"][:,
                                             :self.num_time_step_previous].to(self.device)
            world_future_bev = data["bev"]["bev"][:, self.num_time_step_previous:
                                           self.num_time_step_previous + self.num_time_step_future].to(self.device)

            world_future_bev_predicted_list = []

            ego_previous_location = data["ego"]["location"][:,
                                                            self.num_time_step_previous - 1].to(self.device)
            ego_future_location = data["ego"]["location"][:, self.num_time_step_previous:
                                                          self.num_time_step_previous + self.num_time_step_future].to(self.device)
            ego_future_location_predicted_list = []

            ego_previous_yaw = data["ego"]["rotation"][:,
                                                       self.num_time_step_previous - 1].to(self.device)
            ego_future_yaw = data["ego"]["rotation"][:, self.num_time_step_previous:
                                                     self.num_time_step_previous + self.num_time_step_future].to(self.device)
            ego_future_yaw_predicted_list = []

            ego_previous_speed = data["ego"]["speed"][:,
                                                      self.num_time_step_previous - 1].to(self.device)
            ego_future_speed = data["ego"]["speed"][:, self.num_time_step_previous:
                                                    self.num_time_step_previous + self.num_time_step_future].to(self.device)
            ego_future_speed_predicted_list = []

            ego_future_action = data["ego"]["action"][:, self.num_time_step_previous:
                                                      self.num_time_step_previous + self.num_time_step_future].to(self.device)
            ego_future_action_predicted_list = []

            ego_state_previous = {
                "location": ego_previous_location,
                "yaw": ego_previous_yaw,
                "speed": ego_previous_speed}

            for k in range(self.num_time_step_future):

                # Predict the future bev
                output = self.model(ego_state_previous, world_previous_bev)

                ego_state_next = output["ego_state_next"]
                world_state_next = output["world_state_next"]
                action = output["action"]

                world_future_bev_predicted = F.sigmoid(
                    world_state_next)

                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)
                ego_future_location_predicted_list.append(
                    ego_state_next["location"])
                ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
                ego_future_speed_predicted_list.append(ego_state_next["speed"])
                ego_future_action_predicted_list.append(action)
                # Predict the future ego location

                # Update the previous bev
                world_previous_bev = torch.cat(
                    (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

                ego_state_previous = ego_state_next

            world_future_bev_predicted = torch.stack(
                world_future_bev_predicted_list, dim=1)

            ego_future_location_predicted = torch.stack(
                ego_future_location_predicted_list, dim=1)

            ego_future_yaw_predicted = torch.stack(
                ego_future_yaw_predicted_list, dim=1)

            ego_future_speed_predicted = torch.stack(
                ego_future_speed_predicted_list, dim=1)

            ego_future_action_predicted = torch.stack(
                ego_future_action_predicted_list, dim=1)

            # # Compute the loss
            # if self.reconstruction_loss == F.mse_loss:
            #     loss_reconstruction = self.reconstruction_loss(
            #         world_future_bev_predicted, world_future_bev)
            # else:
            #     # Flatten the predicted and actual values
            #     loss_reconstruction = self.reconstruction_loss(
            #         world_future_bev_predicted.view(
            #             world_future_bev_predicted.shape[0], -1), world_future_bev.view(
            #             world_future_bev.shape[0], -1))

            # Compute the KL divergence
            # loss_kl_div = -0.5 * \
            #     torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # # Compute the total loss
            # loss = loss_reconstruction + loss_kl_div

            cost = self.cost(ego_future_location_predicted, ego_future_yaw_predicted, ego_future_speed_predicted, world_future_bev_predicted)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
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

                world_previous_bev = data["bev"][:,
                                                 :self.num_time_step_previous].to(self.device)
                world_future_bev = data["bev"][:, self.num_time_step_previous:
                                               self.num_time_step_previous + self.num_time_step_future].to(self.device)

                world_future_bev_predicted_list = []
                mu_list = []
                logvar_list = []

                for k in range(self.num_time_step_future):

                    # Predict the future bev
                    world_future_bev_predicted, mu, logvar = self.model(
                        world_previous_bev, world_future_bev[:, k])

                    # if self.reconstruction_loss == F.mse_loss:
                    #     world_future_bev_predicted = F.sigmoid(
                    #         world_future_bev_predicted)
                    world_future_bev_predicted = F.sigmoid(
                        world_future_bev_predicted)

                    world_future_bev_predicted_list.append(
                        world_future_bev_predicted)
                    mu_list.append(mu)
                    logvar_list.append(logvar)

                    # Update the previous bev
                    world_previous_bev = torch.cat(
                        (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

                world_future_bev_predicted = torch.stack(
                    world_future_bev_predicted_list, dim=1)
                mu = torch.stack(mu_list, dim=1)
                logvar = torch.stack(logvar_list, dim=1)

                if self.logvar_clip:
                    logvar = torch.clamp(logvar, self.logvar_clip_min,
                                         self.logvar_clip_max)

                # Calculate the KL divergence loss
                loss_kl_div = -0.5 * \
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Calculate the reconstruction loss
                if self.reconstruction_loss == F.mse_loss:
                    loss_reconstruction = self.reconstruction_loss(
                        world_future_bev_predicted, world_future_bev)
                else:
                    # Flatten the predicted and actual values
                    loss_reconstruction = self.reconstruction_loss(
                        world_future_bev_predicted.view(
                            world_future_bev_predicted.shape[0], -1), world_future_bev.view(
                            world_future_bev.shape[0], -1))

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

        return loss, loss_kl_div, loss_reconstruction

    def learn(self, run=None):

        for epoch in range(self.current_epoch, self.num_epochs):

            self.train(run)
            loss, loss_kl_div, loss_reconstruction = self.validate(run)
            logger.info(
                "Epoch: {}, Val Loss: {}, Val Loss KL Div: {}, Val Loss Reconstruction: {}".format(
                    epoch, loss, loss_kl_div, loss_reconstruction))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if ((epoch + 1) % 5 == 0) and self.save_path is not None:

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "epoch": epoch,
                    "train_step": self.train_step,
                    "val_step": self.val_step},
                    self.save_path /
                    Path(f"checkpoint_{epoch}.pt"))

                run.save(str(self.save_path /
                             Path(f"checkpoint_{epoch}.pt")))
