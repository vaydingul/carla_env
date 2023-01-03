import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.dataset.instance import InstanceDataset
from carla_env.trainer.ego_model import Trainer
from utils.train_utils import (seed_everything, get_device)
from utils.wandb_utils import (create_wandb_run)
from utils.model_utils import (fetch_checkpoint_from_wandb_run)
import wandb
import argparse
import logging
from datetime import datetime
from pathlib import Path
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(config):

    seed_everything(config.seed)
    device = get_device()
    run = create_wandb_run(config)

    dataset_train = InstanceDataset(
        data_path=config.data_path_train,
        sequence_length=config.num_time_step_previous +
        config.num_time_step_future,
        read_keys=["ego"],
        dilation=config.dataset_dilation)
    dataset_val = InstanceDataset(
        data_path=config.data_path_val,
        sequence_length=config.num_time_step_previous +
        config.num_time_step_future,
        read_keys=["ego"],
        dilation=config.dataset_dilation)

    logger.info(f"Train dataset size: {len(dataset_train)}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    if not config.resume:

        model = KinematicBicycleModel(dt=config.dt)

    else:

        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run, checkpoint_number=config.resume_checkpoint_number)

        model = KinematicBicycleModel.load_model_from_wandb_run(
            run=run, checkpoint=checkpoint, device=device)

    model.to(device)

    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if not config.resume:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr)

    else:
        checkpoint = torch.load(
            checkpoint.name,
            map_location=device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=run.config["lr"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])

    loss_criterion = torch.nn.L1Loss()

    if run is not None:
        run.watch(model, log="all")

    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        device=device,
        save_path=config.pretrained_model_path,
        num_time_step_previous=config.num_time_step_previous,
        num_time_step_future=config.num_time_step_future,
        current_epoch=checkpoint["epoch"] + 1 if config.resume else 0,
        num_epochs=config.num_epochs,
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0)

    logger.info("Training started!")

    trainer.learn(run)

    logger.info("Training finished!")


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime('%Y-%m-%d'))
    time_ = Path(datetime.today().strftime('%H-%M-%S'))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_path_train", type=str,
                        default="/home/vaydingul/Documents/Codes/carla_env/data/kinematic_model_train_data_10Hz")
    parser.add_argument("--data_path_val", type=str,
                        default="/home/vaydingul/Documents/Codes/carla_env/data/kinematic_model_val_data_10Hz")
    parser.add_argument("--num_time_step_previous", type=int, default=1)
    parser.add_argument("--num_time_step_future", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1 / 5)
    parser.add_argument("--dataset_dilation", type=int, default=1)
    parser.add_argument("--pretrained_model_path",
                        type=str, default=checkpoint_path)

    # WANDB RELATED PARAMETERS

    parser.add_argument(
        "--resume",
        type=lambda x: (
            str(x).lower() == 'true'),
        default=False)
    parser.add_argument("--resume_checkpoint_number", type=int, default=49)

    parser.add_argument(
        "--wandb",
        type=lambda x: (
            str(x).lower() == 'true'),
        default=False)
    parser.add_argument("--wandb_project", type=str, default="mbl")
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="world-forward-model-multi-step")
    parser.add_argument("--wandb_name", type=str, default="model")
    parser.add_argument("--wandb_id", type=str, default=None)

    config = parser.parse_args()

    main(config)
