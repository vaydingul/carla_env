import logging
import os
import numpy as np
import argparse
from collections import deque
from utilities.kinematic_utils import acceleration_to_throttle_brake
from utilities.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_run_from_wandb_link,
)
from utilities.path_utils import create_date_time_path
from utilities.train_utils import seed_everything, get_device

from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.config_utils import parse_yml, copy_config_to_path
from utilities.log_utils import get_logger, configure_logger, pretty_print_config
from leaderboard.leaderboard_evaluator_local import main as leaderboard_evaluator_main


def main(config):
    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    seed_everything(config["seed"])
    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    pretty_print_config(logger, config)

    # ---------------------------------------------------------------------------- #
    #                                    DEVICE                                    #
    # ---------------------------------------------------------------------------- #
    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN                                  #
    # ---------------------------------------------------------------------------- #
    run = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                    SENSORS                                   #
    # ---------------------------------------------------------------------------- #

    sensor_class_list = sensor_factory(config)
    config["environment"].update({"sensors": sensor_class_list})
    # ---------------------------------------------------------------------------- #
    #                                  ENVIRONMENT                                 #
    # ---------------------------------------------------------------------------- #

    environment_class = environment_factory(config)
    environment = environment_class(config=config["environment"])
    
    copy_config_to_path(config, os.path.join(environment.renderer_module.save_path, "config.yml"))

    # ---------------------------------------------------------------------------- #
    #                    EGO FORWARD MODEL WANDB RUN CHECKPOINT                    #
    # ---------------------------------------------------------------------------- #

    ego_forward_model_wandb_link = config["wandb_ego_forward_model"]["link"]
    ego_forward_model_checkpoint_number = config["wandb_ego_forward_model"][
        "checkpoint_number"
    ]

    ego_forward_model_run = fetch_run_from_wandb_link(ego_forward_model_wandb_link)
    ego_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
        run=ego_forward_model_run,
        checkpoint_number=ego_forward_model_checkpoint_number,
    )
    ego_forward_model_checkpoint_path = ego_forward_model_checkpoint_object.name

    # Create the model
    ego_forward_model_class = ego_forward_model_factory(ego_forward_model_run.config)
    # Initialize the model
    ego_forward_model = ego_forward_model_class.load_model_from_wandb_run(
        config=ego_forward_model_run.config["ego_forward_model"]["config"],
        checkpoint_path=ego_forward_model_checkpoint_path,
        device=device,
    )

    # ---------------------------------------------------------------------------- #
    #                    WORLD FORWARD MODEL WANDB RUN CHECKPOINT                  #
    # ---------------------------------------------------------------------------- #
    if config["wandb_world_forward_model"] is not None:
        world_forward_model_wandb_link = config["wandb_world_forward_model"]["link"]
        world_forward_model_checkpoint_number = config["wandb_world_forward_model"][
            "checkpoint_number"
        ]

        world_forward_model_run = fetch_run_from_wandb_link(
            world_forward_model_wandb_link
        )
        world_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=world_forward_model_run,
            checkpoint_number=world_forward_model_checkpoint_number,
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
        world_forward_model = None

    # ---------------------------------------------------------------------------- #
    #                                   COST                                       #
    # ---------------------------------------------------------------------------- #

    cost_class = cost_factory(config)
    cost = cost_class(device, config["cost"]["config"])

    # ---------------------------------------------------------------------------- #
    #                                   OPTIMIZER                                  #
    # ---------------------------------------------------------------------------- #

    optimizer_class = optimizer_factory(config)

    # ---------------------------------------------------------------------------- #
    #                                    EVALUATOR                                 #
    # ---------------------------------------------------------------------------- #
    evaluator_class = evaluator_factory(config)
    evaluator = evaluator_class(
        environment,
        ego_forward_model,
        world_forward_model,
        cost,
        config["evaluator"]["cost_weight"],
        device,
        optimizer_class,
        config["training"]["optimizer"]["config"],
        batch_size=config["evaluator"]["batch_size"],
        action_size=config["evaluator"]["action_size"],
        num_optimization_iteration_grad=config["evaluator"][
            "num_optimization_iteration_grad"
        ],
        num_optimization_iteration_cem=config["evaluator"][
            "num_optimization_iteration_cem"
        ],
        topk=config["evaluator"]["topk"],
        init_action=config["evaluator"]["init_action"],
        init_action_mean=config["evaluator"]["init_action_mean"],
        init_action_std=config["evaluator"]["init_action_std"],
        skip_frames=config["evaluator"]["skip_frames"],
        repeat_frames=config["evaluator"]["repeat_frames"],
        gradient_clip=config["training"]["gradient_clip"]["enable"],
        gradient_clip_type=config["training"]["gradient_clip"]["type"],
        gradient_clip_value=config["training"]["gradient_clip"]["value"],
        log_video=config["evaluator"]["log_video"],
        log_video_scale=config["evaluator"]["log_video_scale"],
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["evaluator"]["rollout_length"],
        bev_agent_channel=config["evaluator"]["bev_agent_channel"],
        bev_vehicle_channel=config["evaluator"]["bev_vehicle_channel"],
        bev_selected_channels=config["evaluator"]["bev_selected_channels"],
        bev_calculate_offroad=config["evaluator"]["bev_calculate_offroad"],
    )

    leaderboard_evaluator_main(
        config=config["leaderboard"],
        device=device,
        environment=environment,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path_leaderboard",
        type=str,
        default="configs/gradcem/evaluation/config_leaderboard.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config_leaderboard = parse_yml(args.config_path_leaderboard)

    config_leaderboard["config_path_leaderboard"] = args.config_path_leaderboard
    config_leaderboard["leaderboard"]["agent_config"] = None

    # Create structured folder path for checkpoint path based on the
    # experiment_name, checkpoint_name, and date time information
    checkpoint_name = config_leaderboard["leaderboard"]["checkpoint"]
    checkpoint_path = create_date_time_path(
        os.path.join(
            "leaderboard_results",
            config_leaderboard["experiment_type"],
        )
    )
    config_leaderboard["leaderboard"]["checkpoint"] = str(
        checkpoint_path / checkpoint_name
    )
    main(config_leaderboard)
