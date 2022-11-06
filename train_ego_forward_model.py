import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from carla_env.models.dynamic import vehicle as v
from carla_env.dataset.ego_model import EgoModelDataset, EgoModelDatasetV2
from carla_env.trainer.ego_model import Trainer
import wandb
import argparse
import logging
from datetime import datetime
from pathlib import Path
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(config):

    data_path_train = config.data_path_train
    data_path_val = config.data_path_val
    ego_model_dataset_train = EgoModelDatasetV2(data_path_train)
    ego_model_dataset_val = EgoModelDatasetV2(data_path_val)
    logger.info(f"Train dataset size: {len(ego_model_dataset_train)}")
    logger.info(f"Validation dataset size: {len(ego_model_dataset_val)}")

    ego_model_dataloader_train = DataLoader(
        ego_model_dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    ego_model_dataloader_val = DataLoader(
        ego_model_dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    ego_model = v.KinematicBicycleModelV2(dt=1 / 20)
    ego_model_optimizer = torch.optim.Adam(
        ego_model.parameters(), lr=config.lr)
    ego_model_loss_criterion = torch.nn.L1Loss()
    ego_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(ego_model_device)
    ego_model.to(ego_model_device)

    ego_model_trainer = Trainer(
        ego_model,
        ego_model_dataloader_train,
        ego_model_dataloader_val,
        ego_model_optimizer,
        ego_model_loss_criterion,
        ego_model_device,
        num_epochs=config.num_epochs)

    if config.wandb:
        run = wandb.init(project="mbl", group="ego-forward-model",
                         name="training_new_model", config=config)
        run.define_metric("train/step")
        run.define_metric("val/step")
        run.define_metric("model/step")
        run.define_metric(name="train/*", step_metric="train/step")
        run.define_metric(name="val/*", step_metric="val/step")
        run.define_metric(name="model/*", step_metric="model/step")
        run.watch(ego_model)

    else:
        run = None

    ego_model_trainer.learn(run)

    ego_model.to("cpu")

    torch.save(ego_model.state_dict(),
               config.pretrained_model_path / Path("ego_model_new.pt"))


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime('%Y-%m-%d'))
    time_ = Path(datetime.today().strftime('%H-%M-%S'))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_path_train", type=str,
                        default="./data/kinematic_model_data_train_3/")
    parser.add_argument("--data_path_val", type=str,
                        default="./data/kinematic_model_data_val_3/")
    parser.add_argument("--pretrained_model_path",
                        type=str, default=checkpoint_path)
    parser.add_argument("--wandb", type=bool, default=True)
    config = parser.parse_args()

    # config.wandb = False

    main(config)
