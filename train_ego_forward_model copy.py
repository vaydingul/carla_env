import torch
from torch.utils.data import DataLoader


import argparse
import logging
from pathlib import Path

from utils.train_utils import seed_everything, get_device
from utils.wandb_utils import create_wandb_run
from utils.model_utils import fetch_checkpoint_from_wandb_run
from utils.path_utils import create_date_time_path
from utils.config_utils import parse_yml
from utils.log_utils import get_logger, configure_logger
from utils.factory import *


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)

    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    seed_everything(config["seed"])

    # ---------------------------------------------------------------------------- #
    #                                    DEVICE                                    #
    # ---------------------------------------------------------------------------- #
    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN                                  #
    # ---------------------------------------------------------------------------- #
    run = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset = dataset_factory(config)

    # ---------------------------------------------------------------------------- #
    #                         TRAIN AND VALIDATION DATASETS                        #
    # ---------------------------------------------------------------------------- #
    dataset_train = dataset(config["dataset_train"])
    dataset_val = dataset(config["dataset_val"])

    # --------------------- Log information about the dataset -------------------- #
    logger.info(f"Train dataset size: {len(dataset_train)}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")

    # ---------------------------------------------------------------------------- #
    #                       TRAIN AND VALIDATION DATALOADERS                       #
    # ---------------------------------------------------------------------------- #
    dataloader_train = DataLoader(
        dataset_train,
        **config["dataloader_train"],
    )

    dataloader_val = DataLoader(
        dataset_val,
        **config["dataloader_val"],
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Train dataloader size: {len(dataloader_train)}")
    logger.info(f"Validation dataloader size: {len(dataloader_val)}")

    # ---------------------------------------------------------------------------- #
    #                               EGO FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #

    if not config["wandb"]["resume"]:
        # Create the model
        model = ego_forward_model_factory(config)
        # Initialize the model
        model(config["ego_forward_model"])

    else:

        # Fetch the specific checkpoint from wandb cloud storage
        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run, checkpoint_number=config.resume_checkpoint_number
        )

        # Create and initialize the model with pretrained weights and biases
        model = model.load_model_from_wandb_run(
            config=run.config["ego_forward_model"],
            checkpoint=checkpoint.name,
            device=device,
        )

    model.to(device)

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if not config.resume:
        optimizer = optimizer_factory(config)
        optimizer = optimizer_factory(model.parameters(), lr=config.lr)

    else:

        checkpoint = torch.load(checkpoint.name, map_location=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=run.config["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss_criterion = loss_criterion_factory(config)

    if run is not None:
        run.watch(model, log="all")

    # ---------------------------------------------------------------------------- #
    #                                    TRAINER                                   #
    # ---------------------------------------------------------------------------- #
    trainer = trainer_factory(config)

    trainer = trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        device=device,
        save_path=config["checkpoint_path"],
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["num_time_step_future"],
        current_epoch=checkpoint["epoch"] + 1 if config.resume else 0,
        num_epochs=config["training"]["num_epochs"],
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0,
    )

    logger.info("Training started!")

    trainer.learn(run)

    logger.info("Training finished!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/ego_forward_model/training/config.yml"
    )
    config = parser.parse_args()

    config = parse_yml(config.config)
    config["checkpoint_path"] = create_date_time_path(config["checkpoint_path"])

    main(config)
