import argparse
import logging
import torch

import os

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader


from carla_env.sampler.distributed_weighted_sampler import DistributedWeightedSampler
from utils.train_utils import seed_everything
from utils.wandb_utils import create_wandb_run
from utils.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_checkpoint_from_wandb_link,
)
from utils.path_utils import create_date_time_path
from utils.config_utils import parse_yml
from utils.log_utils import get_logger, configure_logger, pretty_print_config
from utils.factory import *


def ddp_setup(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, config):

    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    if rank == 0:
        pretty_print_config(logger, config)

    # ---------------------------- TORCH related stuff --------------------------- #
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------- #
    #                                   DDP SETUP                                  #
    # ---------------------------------------------------------------------------- #
    ddp_setup(rank, world_size, config["training"]["master_port"])

    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    seed_everything(config["seed"])

    # ---------------------------------------------------------------------------- #
    #                                    DEVICE                                    #
    # ---------------------------------------------------------------------------- #
    device = rank  # get_device()

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN                                  #
    # ---------------------------------------------------------------------------- #
    run = create_wandb_run(config if rank == 0 else None)

    if config["wandb"]["resume"]:
        # Fetch the specific checkpoint from wandb cloud storage
        checkpoint_object = fetch_checkpoint_from_wandb_link(
            wandb_link=config["wandb"]["link"],
            checkpoint_number=config["wandb"]["resume_checkpoint_number"],
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

    # ----------------------------- Weighted Sampling ---------------------------- #
    if config["training"]["weighted_sampling"]:

        logger.info(f"Weighted sampling is {config.weighted_sampling}")

        weights = torch.load(
            f"{config['dataset_train']['data_path']}/weights_{config['num_time_step_previous']}_{config['num_time_step_future']}_{config['dataset_train']['dilation']}.pt"
        )

    else:

        weights = None

    # ---------------------------------------------------------------------------- #
    #                       TRAIN AND VALIDATION DATALOADERS                       #
    # ---------------------------------------------------------------------------- #
    dataloader_train = DataLoader(
        dataset_train,
        **config["dataloader_train"],
        sampler=DistributedWeightedSampler(
            dataset_train,
            weights=weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ),
    )

    dataloader_val = DataLoader(
        dataset_val,
        **config["dataloader_val"],
        sampler=DistributedWeightedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        ),
    )

    dataloader_val = DataLoader(dataset_val, **config["dataloader_val"])

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Train dataloader size: {len(dataloader_train)}")
    logger.info(f"Validation dataloader size: {len(dataloader_val)}")

    # ---------------------------------------------------------------------------- #
    #                               WORLD FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #

    if config["wandb"]["resume"]:

        # Create the model
        model_class = world_forward_model_factory(run.config)

        model = model_class.load_model_from_wandb_run(
            config=run.config["world_forward_model"]["config"],
            checkpoint_path=checkpoint_path,
            device=device,
        )

    else:

        model_class = world_forward_model_factory(config)

        # Initialize the model
        model = model_class(config["world_forward_model"]["config"])

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

        if run.config["training"]["scheduler"]["enable"]:
            scheduler_class = scheduler_factory(run.config)
            scheduler = scheduler_class(
                optimizer,
                **run.config["training"]["scheduler"]["config"],
            )

            # Load the scheduler state dictionary

            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        else:

            scheduler = None

    else:

        optimizer_class = optimizer_factory(config)
        optimizer = optimizer_class(
            model.parameters(), **config["training"]["optimizer"]["config"]
        )

        if config["training"]["scheduler"]["enable"]:
            scheduler_class = scheduler_factory(config)
            scheduler = scheduler_class(
                optimizer, **config["training"]["scheduler"]["config"]
            )

        else:

            scheduler = None

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

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    model.to(device)

    # ---------------------------------------------------------------------------- #
    #                                    TRAINER                                   #
    # ---------------------------------------------------------------------------- #
    trainer_class = trainer_factory(config)

    trainer = trainer_class(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        rank=device,
        lr_scheduler=scheduler,
        reconstruction_loss=loss_criterion,
        sigmoid_before_loss=config["training"]["sigmoid_before_loss"],
        save_interval=config["training"]["save_interval"],
        val_interval=config["training"]["val_interval"],
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["num_time_step_future"],
        num_epochs=config["training"]["num_epochs"],
        current_epoch=checkpoint["epoch"] + 1 if config["wandb"]["resume"] else 0,
        logvar_clip=config["training"]["logvar_clip"]["enable"],
        logvar_clip_min=config["training"]["logvar_clip"]["min"],
        logvar_clip_max=config["training"]["logvar_clip"]["max"],
        gradient_clip=config["training"]["gradient_clip"]["enable"],
        gradient_clip_type=config["training"]["gradient_clip"]["type"],
        gradient_clip_value=config["training"]["gradient_clip"]["value"],
        save_path=config["checkpoint_path"],
        train_step=checkpoint["train_step"] if config["wandb"]["resume"] else 0,
        val_step=checkpoint["val_step"] if config["wandb"]["resume"] else 0,
    )

    logger.info("Training started!")
    if rank != 0:
        run = create_wandb_run(None)
    trainer.learn(run)
    logger.info("Training finished!")

    destroy_process_group()

    if run is not None:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/world_forward_model/training/config.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["checkpoint_path"] = create_date_time_path(config["checkpoint_path"])
    config["config_path"] = args.config_path

    mp.spawn(
        main,
        args=(config["training"]["num_gpu"], config),
        nprocs=config["training"]["num_gpu"],
    )
