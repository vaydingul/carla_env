import logging
import numpy as np
import torch
from pathlib import Path
from utils.train_utils import clone, to, cat, stack

torch.autograd.set_detect_anomaly(True)
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
        save_path,
        num_time_step_previous=1,
        num_time_step_future=1,
        current_epoch=0,
        num_epochs=1000,
        save_interval=50,
        train_step=0,
        val_step=0,
    ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.device = device
        self.save_path = save_path
        self.num_time_step_previous = num_time_step_previous
        self.num_time_step_future = num_time_step_future
        self.current_epoch = current_epoch
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.train_step = train_step
        self.val_step = val_step

    def train(self, run):
        self.model.train()

        for i, (data) in enumerate(self.dataloader_train):
            loss_dict = self.shared_step(data, i, mode="train")

            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            run.log(
                {
                    "train/step": self.train_step,
                    "train/loss": loss_dict["loss"],
                    "train/loss_location": loss_dict["loss_location"],
                    "train/loss_rotation": loss_dict["loss_rotation"],
                }
            )

    def validate(self, run=None):
        self.model.eval()

        losses_total = []
        losses_location = []
        losses_rotation = []

        with torch.no_grad():
            for i, (data) in enumerate(self.dataloader_val):
                loss_dict = self.shared_step(data, i, mode="val")

                losses_total.append(loss_dict["loss"].item())
                losses_location.append(loss_dict["loss_location"].item())
                losses_rotation.append(loss_dict["loss_rotation"].item())

        loss = np.mean(losses_total)
        loss_location = np.mean(losses_location)
        loss_rotation = np.mean(losses_rotation)

        run.log(
            {
                "val/step": self.val_step,
                "val/loss": loss,
                "val/loss_location": loss_location,
                "val/loss_rotation": loss_rotation,
            }
        )

        return {
            "loss": loss,
            "loss_location": loss_location,
            "loss_rotation": loss_rotation,
        }

    def shared_step(self, data, i, mode="train"):
        ego_state_previous = to(
            data["ego"],
            self.device,
            index_start=self.num_time_step_previous - 1,
            index_end=self.num_time_step_previous,
        )
        ego_state_previous["rotation_array"].deg2rad_()

        ego_state_future = to(
            data["ego"],
            self.device,
            index_start=self.num_time_step_previous,
            index_end=self.num_time_step_previous + self.num_time_step_future,
        )
        ego_state_future["rotation_array"].deg2rad_()

        ego_state_future_predicted_list = []

        ego_action = data["ego"]["control_array"][:, : self.num_time_step_future].to(
            self.device
        )

        ego_action[..., 0] -= ego_action[..., -1]
        ego_action = ego_action[..., :-1]

        for t in range(self.num_time_step_future):
            control_ = ego_action[:, t : t + 1]

            ego_state_next = self.model(ego_state_previous, control_)
            ego_state_future_predicted_list.append(ego_state_next)
            ego_state_previous = ego_state_next

        ego_state_future_predicted = cat(ego_state_future_predicted_list, dim=1)

        ego_future_location = ego_state_future["location_array"][..., :2]
        ego_future_location_predicted = ego_state_future_predicted["location_array"][
            ..., :2
        ]

        ego_future_yaw = ego_state_future["rotation_array"][..., 2:3]
        ego_future_yaw_predicted = ego_state_future_predicted["rotation_array"][
            ..., 2:3
        ]

        loss_location = self.loss_criterion(
            ego_future_location, ego_future_location_predicted
        )
        loss_rotation = self.loss_criterion(
            torch.cos(ego_future_yaw), torch.cos(ego_future_yaw_predicted)
        )
        loss_rotation += self.loss_criterion(
            torch.sin(ego_future_yaw), torch.sin(ego_future_yaw_predicted)
        )

        loss = loss_location + loss_rotation

        if mode == "train":
            self.train_step += ego_future_location.shape[0]
        elif mode == "val":
            self.val_step += ego_future_location.shape[0]
        else:
            raise ValueError("mode should be either train or val")

        return {
            "loss": loss,
            "loss_location": loss_location,
            "loss_rotation": loss_rotation,
        }

    def learn(self, run=None):
        loss_dict = self.validate(run)
        logger.info(f"{'=' * 20}")

        logger.info(f"Epoch: Start")
        for k, v in loss_dict.items():
            logger.info(f"Val {k}: {v}")
        logger.info(f"{'=' * 20}")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch = epoch
            self.train(run)
            loss_dict = self.validate(run)

            logger.info(f"{'=' * 20}")
            logger.info(f"Epoch: {epoch}")
            for k, v in loss_dict.items():
                logger.info(f"Val {k}: {v}")
            logger.info(f"{'=' * 20}")

            if ((epoch + 1) % self.save_interval == 0) and self.save_path is not None:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "train_step": self.train_step,
                        "val_step": self.val_step,
                    },
                    self.save_path / Path(f"checkpoint_{epoch}.pt"),
                )

                if run is not None:
                    run.save(
                        str(self.save_path / Path(f"checkpoint_{epoch}.pt")),
                    )
