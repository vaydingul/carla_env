import logging
import os
import argparse

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier


from utils.train_utils import seed_everything, organize_device
from utils.wandb_utils import create_wandb_run, DummyWandb
from utils.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_run_from_wandb_link,
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
    policy_model_run = create_wandb_run(config if rank == 0 else None)

    if config["wandb"]["resume"]:
        # Fetch the specific checkpoint from wandb cloud storage
        policy_model_checkpoint_object = fetch_checkpoint_from_wandb_link(
            wandb_link=config["wandb"]["link"],
            checkpoint_number=config["wandb"]["resume_checkpoint_number"],
        )
        policy_model_checkpoint_path = policy_model_checkpoint_object.name
        policy_model_checkpoint = torch.load(
            f=policy_model_checkpoint_path, map_location=organize_device(device)
        )

        policy_model_run_ = fetch_run_from_wandb_link(config["wandb"]["link"])

    # ---------------------------------------------------------------------------- #
    #                    EGO FORWARD MODEL WANDB RUN CHECKPOINT                    #
    # ---------------------------------------------------------------------------- #

    if config["wandb_ego_forward_model"]["enable"] is not None:

        ego_forward_model_run = fetch_run_from_wandb_link(
            config["wandb_ego_forward_model"]["link"]
        )
        ego_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=ego_forward_model_run,
            checkpoint_number=config["wandb_ego_forward_model"]["checkpoint_number"],
        )
        ego_forward_model_checkpoint_path = ego_forward_model_checkpoint_object.name

        # Create the model
        ego_forward_model_class = ego_forward_model_factory(
            ego_forward_model_run.config
        )
        # Initialize the model
        ego_forward_model = ego_forward_model_class.load_model_from_wandb_run(
            config=ego_forward_model_run.config["ego_forward_model"]["config"],
            checkpoint_path=ego_forward_model_checkpoint_path,
            device=device,
        )

    else:

        # Create the model
        ego_forward_model_class = ego_forward_model_factory(config)
        # Initialize the model
        ego_forward_model = ego_forward_model_class(
            config["ego_forward_model"]["config"]
        )

    # ---------------------------------------------------------------------------- #
    #                   WORLD FORWARD MODEL WANDB RUN CHECKPOINT                   #
    # ---------------------------------------------------------------------------- #
    if config["wandb_world_forward_model"]["enable"] is not None:

        world_forward_model_run = fetch_run_from_wandb_link(
            config["wandb_world_forward_model"]["link"]
        )
        world_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=world_forward_model_run,
            checkpoint_number=config["wandb_world_forward_model"]["checkpoint_number"],
        )
        world_forward_model_checkpoint_path = world_forward_model_checkpoint_object.name

        # Create the model
        world_forward_model_class = world_forward_model_factory(
            world_forward_model_run.config
        )

        # Initialize the model
        world_forward_model = world_forward_model_class.load_model_from_wandb_run(
            config=world_forward_model_run.config["world_forward_model"]["config"],
            checkpoint_path=world_forward_model_checkpoint_path,
            device=device,
        )

    else:

        # Create the model
        world_forward_model_class = world_forward_model_factory(config)
        # Initialize the model
        world_forward_model = world_forward_model_class(
            config["world_forward_model"]["config"]
        )

    # ---------------------------------------------------------------------------- #
    #                                 POLICY MODEL                                 #
    # ---------------------------------------------------------------------------- #

    if config["wandb"]["resume"]:

        policy_model_class = policy_model_factory(policy_model_run_.config)

        # Create and initialize the model with pretrained weights and biases
        policy_model = policy_model_class.load_model_from_wandb_run(
            config=policy_model_run_.config["policy_model"]["config"],
            checkpoint_path=policy_model_checkpoint_path,
            device=device,
        )

    else:

        if config["training"]["use_world_forward_model_encoder_output_as_world_state"]:

            config["policy_model"]["config"]["input_shape_world_state"] = tuple(
                world_forward_model.world_previous_bev_encoder.get_output_shape()
            )

        else:

            config["policy_model"]["config"]["input_shape_world_state"] = tuple(
                world_forward_model.get_input_shape_previous()
            )

        policy_model_run.config.update({"policy_model": config["policy_model"]}, allow_val_change=True)
        

        policy_model_class = policy_model_factory(config)
        # Create and initialize the model
        policy_model = policy_model_class(
            config["policy_model"]["config"],
        )

    ego_forward_model.to(device)
    world_forward_model.to(device)
    policy_model.to(device)

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
        sampler=DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config["seed"],
        ),
    )

    dataloader_val = DataLoader(
        dataset_val,
        **config["dataloader_val"],
        sampler=DistributedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=config["seed"],
        ),
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Train dataloader size: {len(dataloader_train)}")
    logger.info(f"Validation dataloader size: {len(dataloader_val)}")

    # ---------------------------------------------------------------------------- #
    #                                   COST                                       #
    # ---------------------------------------------------------------------------- #

    cost_class = cost_factory(config)
    cost = cost_class(device, config["cost"]["config"])
    # cost.to(device)
    # ---------------------------------------------------------------------------- #
    #                            OPTIMIZER AND SCHEDULER                           #
    # ---------------------------------------------------------------------------- #

    optimization_parameters = []

    for param in config["training"]["optimizer"]["parameters"]:

        if param == "ego":

            optimization_parameters.append({"params": ego_forward_model.parameters()})

        elif param == "world":

            optimization_parameters.append({"params": world_forward_model.parameters()})

        elif param == "policy":

            optimization_parameters.append({"params": policy_model.parameters()})

        else:

            raise ValueError(f"Optimizer parameter {param} not recognized")

    if config["wandb"]["resume"]:

        optimizer_class = optimizer_factory(policy_model_run_.config)
        optimizer = optimizer_class(
            optimization_parameters,
            **policy_model_run_.config["training"]["optimizer"]["config"],
        )
        # Load the optimizer state dictionary
        optimizer.load_state_dict(policy_model_checkpoint["optimizer_state_dict"])

        if policy_model_run_.config["training"]["scheduler"]["enable"]:
            scheduler_class = scheduler_factory(policy_model_run_.config)
            scheduler = scheduler_class(
                optimizer,
                **policy_model_run_.config["training"]["scheduler"]["config"],
            )

            # Load the scheduler state dictionary

            scheduler.load_state_dict(policy_model_checkpoint["scheduler_state_dict"])

        else:

            scheduler = None

    else:

        optimizer_class = optimizer_factory(config)
        optimizer = optimizer_class(
            optimization_parameters, **config["training"]["optimizer"]["config"]
        )

        if config["training"]["scheduler"]["enable"]:
            scheduler_class = scheduler_factory(config)
            scheduler = scheduler_class(
                optimizer, **config["training"]["scheduler"]["config"]
            )

        else:

            scheduler = None

    policy_model_run.save(config["config_path"])
    policy_model_run.watch(policy_model, log="all")

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Ego forward model number of parameters: {sum(p.numel() for p in ego_forward_model.parameters() if p.requires_grad)}"
    )
    logger.info(
        f"World forward model number of parameters: {sum(p.numel() for p in world_forward_model.parameters() if p.requires_grad)}"
    )
    logger.info(
        f"Policy model number of parameters: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad)}"
    )

    # ---------------------------------------------------------------------------- #
    #                                    TRAINER                                   #
    # ---------------------------------------------------------------------------- #
    trainer_class = trainer_factory(config)

    trainer = trainer_class(
        ego_forward_model=ego_forward_model,
        world_forward_model=world_forward_model,
        policy_model=policy_model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        rank=device,
        cost=cost,
        cost_weight=config["training"]["cost_weight"],
        lr_scheduler=scheduler,
        save_interval=config["training"]["save_interval"],
        val_interval=config["training"]["val_interval"],
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["num_time_step_future"],
        num_epochs=config["training"]["num_epochs"],
        current_epoch=policy_model_checkpoint["epoch"] + 1
        if config["wandb"]["resume"]
        else 0,
        gradient_clip=config["training"]["gradient_clip"]["enable"],
        gradient_clip_type=config["training"]["gradient_clip"]["type"],
        gradient_clip_value=config["training"]["gradient_clip"]["value"],
        binary_occupancy=config["training"]["binary_occupancy"]["enable"],
        binary_occupancy_threshold=config["training"]["binary_occupancy"]["threshold"],
        use_ground_truth=config["training"]["use_ground_truth"],
        use_world_forward_model_encoder_output_as_world_state=config["training"][
            "use_world_forward_model_encoder_output_as_world_state"
        ],
        debug_render=config["training"]["debug_render"] if rank == 0 else False,
        renderer=config["training"]["renderer"],
        bev_agent_channel=config["bev_agent_channel"],
        bev_vehicle_channel=config["bev_vehicle_channel"],
        bev_selected_channels=config["bev_selected_channels"],
        bev_calculate_offroad=config["bev_calculate_offroad"],
        save_path=config["checkpoint_path"] if rank == 0 else None,
        train_step=policy_model_checkpoint["train_step"]
        if config["wandb"]["resume"]
        else 0,
        val_step=policy_model_checkpoint["val_step"]
        if config["wandb"]["resume"]
        else 0,
    )

    logger.info("Training started!")
    trainer.learn(policy_model_run if rank == 0 else DummyWandb())
    logger.info("Training finished!")

    destroy_process_group()

    policy_model_run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/policy_model/training/config_overfit_simple_bev.yml",
    )

    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["checkpoint_path"] = create_date_time_path(config["checkpoint_path"])

    config["config_path"] = args.config_path

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN                                  #
    # ---------------------------------------------------------------------------- #
    # policy_model_run = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    # dataset_class = dataset_factory(config)

    # ---------------------------------------------------------------------------- #
    #                         TRAIN AND VALIDATION DATASETS                        #
    # ---------------------------------------------------------------------------- #
    # dataset_train = dataset_class(config["dataset_train"]["config"])
    # dataset_val = dataset_class(config["dataset_val"]["config"])

    mp.spawn(
        main,
        args=(
            config["training"]["num_gpu"],
            config,
            # policy_model_run,
            # dataset_train,
            # dataset_val,
        ),
        nprocs=config["training"]["num_gpu"],
    )
