import torch
from torch.utils.data import DataLoader


import argparse
import logging
from pathlib import Path

from utilities.train_utils import seed_everything, get_device
from utilities.wandb_utils import create_wandb_run
from utilities.model_utils import fetch_checkpoint_from_wandb_run
from utilities.path_utils import create_date_time_path
from utilities.config_utils import parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config
from utilities.factory import *


def main(config):
    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    pretty_print_config(logger, config)

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
    if config["wandb"]["resume"]:
        # Fetch the specific checkpoint from wandb cloud storage
        checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=run, checkpoint_number=config.resume_checkpoint_number
        )
        checkpoint_path = checkpoint_object.name
        checkpoint = torch.load(f=checkpoint_path, map_location=device)
    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(config)

    # ---------------------------------------------------------------------------- #
    #                         TRAIN AND VALIDATION DATASETS                        #
    # ---------------------------------------------------------------------------- #
    dataset_train = dataset_class(config["dataset_train"]["config"])
    dataset_val = dataset_class(config["dataset_val"]["config"])

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
    #                               WORLD FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #

    if config["wandb"]["resume"]:
        # Create the model
        model_class = ego_forward_model_factory(run.config)

        model = model_class.load_model_from_wandb_run(
            config=run.config["ego_forward_model"]["config"],
            checkpoint_path=checkpoint_path,
            device=device,
        )

    else:
        model_class = ego_forward_model_factory(config)

        # Initialize the model
        model = model_class(config["ego_forward_model"]["config"])

    # ---------------------------------------------------------------------------- #
    #                            OPTIMIZER AND SCHEDULER                           #
    # ---------------------------------------------------------------------------- #

    if config["wandb"]["resume"]:
        optimizer_class = optimizer_factory(run.config)
        optimizer = optimizer_class(
            model.parameters(), **run.config["optimizer"]["config"]
        )
        # Load the optimizer state dictionary
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        optimizer_class = optimizer_factory(config)
        optimizer = optimizer_class(
            model.parameters(), **config["training"]["optimizer"]["config"]
        )

    model.to(device)

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    loss_criterion_class = loss_criterion_factory(config)

    # ---------------------------------------------------------------------------- #
    #                                LOSS CRITERION                                #
    # ---------------------------------------------------------------------------- #
    loss_criterion_class = loss_criterion_factory(config)

    # The stupidest thing I've ever done
    if "pos_weight" in config["training"]["loss"]["config"].keys():
        config["training"]["loss"]["config"]["pos_weight"] = torch.tensor(
            config["training"]["loss"]["config"]["pos_weight"]
        ).to(device)

    if "weight" in config["training"]["loss"]["config"].keys():
        config["training"]["loss"]["config"]["weight"] = torch.tensor(
            config["training"]["loss"]["config"]["weight"]
        ).to(device)

    loss_criterion = loss_criterion_class(**config["training"]["loss"]["config"])

    if run is not None:
        run.save(config["config_path"])
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
        current_epoch=checkpoint["epoch"] + 1 if config["wandb"]["resume"] else 0,
        num_epochs=config["training"]["num_epochs"],
        save_interval=config["training"]["save_interval"],
        train_step=checkpoint["train_step"] if config["wandb"]["resume"] else 0,
        val_step=checkpoint["val_step"] if config["wandb"]["resume"] else 0,
    )

    logger.info("Training started!")

    trainer.learn(run)

    logger.info("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/ego_forward_model/training/kinematic/config_20Hz_action_repeat_4.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["checkpoint_path"] = create_date_time_path(config["checkpoint_path"])
    config["config_path"] = args.config_path

    main(config)
