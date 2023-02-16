import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.evaluator.world_forward_model import Evaluator
from utils.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_checkpoint_from_wandb_link,
)

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.evaluator.ego_forward_model import Evaluator
from utils.train_utils import get_device, seed_everything
from utils.model_utils import fetch_run_from_wandb_link, fetch_checkpoint_from_wandb_run
from utils.path_utils import create_date_time_path
from utils.config_utils import parse_yml
from utils.factory import *
from utils.wandb_utils import create_wandb_run
from utils.log_utils import get_logger, configure_logger, pretty_print_config
import wandb


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
    #                                   WANDB RUN CHECKPOINT                                #
    # ---------------------------------------------------------------------------- #
    run = fetch_run_from_wandb_link(config["wandb"]["link"])
    checkpoint_object = fetch_checkpoint_from_wandb_run(
        run=run, checkpoint_number=config["wandb"]["checkpoint_number"]
    )
    checkpoint_path = checkpoint_object.name
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    # ---------------------------------------------------------------------------- #
    #                                     WANDB                                    #
    # ---------------------------------------------------------------------------- #
    run_eval = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(run.config)

    # ---------------------------------------------------------------------------- #
    #                         TEST DATASET                                         #
    # ---------------------------------------------------------------------------- #
    dataset_test = dataset_class(config["dataset_test"])

    # --------------------- Log information about the dataset -------------------- #
    logger.info(f"Test dataset size: {len(dataset_test)}")

    # ---------------------------------------------------------------------------- #
    #                       TEST DATALOADER                                        #
    # ---------------------------------------------------------------------------- #
    dataloader_test = DataLoader(
        Subset(
            dataset_test,
            range(
                0,
                len(dataset_test),
                config["evaluation"]["test_step"],
            ),
        ),
        **config["dataloader_test"],
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Test dataloader size: {len(dataloader_test)}")

    # ---------------------------------------------------------------------------- #
    #                               WORLD FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #
    model_class = world_forward_model_factory(run.config)

    # Create and initialize the model with pretrained weights and biases
    model = model_class.load_model_from_wandb_run(
        config=run.config["world_forward_model"],
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.to(device)

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # ---------------------------------------------------------------------------- #
    #                                   EVALUATOR                                  #
    # ---------------------------------------------------------------------------- #
    evaluator_class = evaluator_factory(config)

    evaluator = evaluator_class(
        model=model,
        dataset=dataset_test,
        dataloader=dataloader_test,
        device=device,
        renderer=config["evaluation"]["renderer"],
        metrics=config["evaluation"]["metrics"],
        num_time_step_previous=config["evaluation"]["num_time_step_previous"],
        num_time_step_predict=config["evaluation"]["num_time_step_predict"],
        thresholds=config["evaluation"]["thresholds"],
        wandb_log_interval=config["evaluation"]["wandb_log_interval"],
        save_path=f"{config['save_path']}",
    )

    logger.info(f"Starting evaluation")

    evaluator.evaluate(run_eval)

    logger.info(f"Finished evaluation")

    # evaluator = Evaluator(
    #     model=model,
    #     dataloader=dataloader_test,
    #     device=device,
    #     report_metrics=config.report_metrics,
    #     metrics=config.metrics,
    #     num_time_step_previous=run.config["num_time_step_previous"],
    #     num_time_step_predict=(
    #         config.num_time_step_predict
    #         if config.num_time_step_predict > 0
    #         else run.config["num_time_step_future"]
    #     ),
    #     threshold=config.threshold,
    #     vehicle_threshold=config.vehicle_threshold,
    #     bev_selected_channels=config.bev_selected_channels,
    #     save_path=f"{config.save_path}/{run.config['num_time_step_previous']}-{run.config['num_time_step_future']}-{(config.num_time_step_predict if config.num_time_step_predict > 0 else run.config['num_time_step_future'])}-{run.config['reconstruction_loss']}-{config.threshold}-{config.checkpoint_number}",
    # )

    # evaluator.evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/vaydingul/Documents/Codes/carla_env/configs/world_forward_model/evaluation/config.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)

    assert (
        config["dataset_test"]["sequence_length"]
        == config["evaluation"]["sequence_length"]
    ), "Sequence length of the dataset and the evaluator should be the same"

    main(config)
