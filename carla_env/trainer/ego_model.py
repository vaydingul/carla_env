import logging
import numpy as np
import torch
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            loss_criterion,
            device,
            num_epochs=1000,
            log_interval=10):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = log_interval

    def train(self, epoch, run):

        self.model.train()

        counter = 0

        for i, (location, rotation, velocity, _, control,
                _) in enumerate(self.dataloader_train):

            location = location[..., :2].to(self.device)
            rotation = rotation[..., 1:2].to(self.device)
            #velocity = velocity[..., :1].to(self.device)
            velocity = torch.norm(
                velocity, dim=-1, keepdim=True).to(self.device)
            control = control.to(self.device)

            location_pred = []
            rotation_pred = []
            velocity_pred = []

            location_ = location[:, 0]
            rotation_ = rotation[:, 0]
            velocity_ = velocity[:, 0]

            for t in range(location.shape[1] - 1):

                control_ = control[:, t]

                location_, rotation_, velocity_ = self.model(
                    location_, rotation_, velocity_, control_)

                location_pred.append(location_)
                rotation_pred.append(rotation_)
                velocity_pred.append(velocity_)

            location_pred = torch.stack(location_pred, dim=1)
            rotation_pred = torch.stack(rotation_pred, dim=1)
            velocity_pred = torch.stack(velocity_pred, dim=1)

            loss_location = self.loss_criterion(location[:, 1:], location_pred)
            loss_rotation = self.loss_criterion(
                torch.cos(rotation[:, 1:]), torch.cos(rotation_pred))
            loss_rotation += self.loss_criterion(
                torch.sin(rotation[:, 1:]), torch.sin(rotation_pred))

            loss = loss_location + loss_rotation

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.dataloader_train.dataset) + counter * \
                self.dataloader_train.batch_size + location.shape[0]

            if run is not None:
                run.log({"train/step": step,
                         "train/loss": loss,
                         "train/loss_location": loss_location,
                         "train/loss_rotation": loss_rotation})

    def validate(self, epoch, run=None):

        self.model.eval()

        losses_total = []
        losses_location = []
        losses_rotation = []

        counter = 0
        with torch.no_grad():

            for i, (location, rotation, velocity, _, control,
                    _) in enumerate(self.dataloader_val):

                location = location[..., :2].to(self.device)
                rotation = rotation[..., 1:2].to(self.device)
                velocity = torch.norm(
                    velocity, dim=-1, keepdim=True).to(self.device)
                control = control.to(self.device)

                location_pred = []
                rotation_pred = []
                velocity_pred = []

                location_ = location[:, 0]
                rotation_ = rotation[:, 0]
                velocity_ = velocity[:, 0]

                for t in range(location.shape[1] - 1):

                    control_ = control[:, t]

                    location_, rotation_, velocity_ = self.model(
                        location_, rotation_, velocity_, control_)

                    location_pred.append(location_)
                    rotation_pred.append(rotation_)
                    velocity_pred.append(velocity_)

                location_pred = torch.stack(location_pred, dim=1)
                rotation_pred = torch.stack(rotation_pred, dim=1)
                velocity_pred = torch.stack(velocity_pred, dim=1)

                loss_location = self.loss_criterion(
                    location_pred, location[:, 1:])
                loss_rotation = self.loss_criterion(
                    torch.cos(rotation_pred), torch.cos(rotation[:, 1:]))
                loss_rotation += self.loss_criterion(
                    torch.sin(rotation_pred), torch.sin(rotation[:, 1:]))

                loss = loss_location + loss_rotation

                losses_total.append(loss.item())
                losses_location.append(loss_location.item())
                losses_rotation.append(loss_rotation.item())

                step = epoch * len(self.dataloader_val.dataset) + counter * \
                    self.dataloader_val.batch_size + location.shape[0]
                counter += 1

        loss = np.mean(losses_total)
        loss_location = np.mean(losses_location)
        loss_rotation = np.mean(losses_rotation)

        if run is not None:
            run.log({"val/step": step,
                     "val/loss": loss,
                     "val/loss_location": loss_location,
                     "val/loss_rotation": loss_rotation})
            run.log({"model/step": step, **{f"model/{name}": param.item()
                    for name, param in self.model.named_parameters()}})

        return loss, loss_location, loss_rotation

    def learn(self, run=None):

        for epoch in range(self.num_epochs):

            self.train(epoch, run)
            loss, loss_location, loss_orientation = self.validate(epoch, run)
            logger.info(
                "Epoch: {}, Val Loss: {}, Val Loss Location: {}, Val Loss Orientation: {}".format(
                    epoch, loss, loss_location, loss_orientation))
