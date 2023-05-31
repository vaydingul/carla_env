import logging
import numpy as np
import argparse
from collections import deque
from utilities.kinematic_utils import acceleration_to_throttle_brake
from utilities.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_run_from_wandb_link,
)
from utilities.train_utils import seed_everything, get_device

from utilities.path_utils import create_date_time_path
from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.config_utils import parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config


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

    # ---------------------------------------------------------------------------- #
    #                       POLICY MODEL WANDB RUN CHECKPOINT                      #
    # ---------------------------------------------------------------------------- #

    policy_model_run = fetch_run_from_wandb_link(config["wandb_policy_model"]["link"])
    policy_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
        run=policy_model_run,
        checkpoint_number=config["wandb_policy_model"]["checkpoint_number"],
    )
    policy_model_checkpoint_path = policy_model_checkpoint_object.name

    # Create the model
    policy_model_class = policy_model_factory(policy_model_run.config)
    # Initialize the model
    policy_model = policy_model_class.load_model_from_wandb_run(
        config=policy_model_run.config["policy_model"]["config"],
        checkpoint_path=policy_model_checkpoint_path,
        device=device,
    )

    # ---------------------------------------------------------------------------- #
    #                    EGO FORWARD MODEL WANDB RUN CHECKPOINT                    #
    # ---------------------------------------------------------------------------- #

    if "wandb_ego_forward_model" not in config.keys():
        ego_forward_model_wandb_link = policy_model_run.config[
            "wandb_ego_forward_model"
        ]["link"]
        ego_forward_model_checkpoint_number = policy_model_run.config[
            "wandb_ego_forward_model"
        ]["checkpoint_number"]
    else:
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
    #                    WORLD FORWARD MODEL WANDB RUN CHECKPOINT                    #
    # ---------------------------------------------------------------------------- #

    if "wandb_world_forward_model" not in config.keys():
        world_forward_model_wandb_link = policy_model_run.config[
            "wandb_world_forward_model"
        ]["link"]
        world_forward_model_checkpoint_number = policy_model_run.config[
            "wandb_world_forward_model"
        ]["checkpoint_number"]
    else:
        world_forward_model_wandb_link = config["wandb_world_forward_model"]["link"]
        world_forward_model_checkpoint_number = config["wandb_world_forward_model"][
            "checkpoint_number"
        ]

    world_forward_model_run = fetch_run_from_wandb_link(world_forward_model_wandb_link)
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

    # ---------------------------------------------------------------------------- #
    #                                   COST                                       #
    # ---------------------------------------------------------------------------- #

    cost_class = cost_factory(policy_model_run.config)
    cost = cost_class(device, policy_model_run.config["cost"]["config"])

    # ------------------------------- BEV Handling ------------------------------- #
    if "bev_agent_channel" in config["tester"]:
        bev_agent_channel = config["tester"]["bev_agent_channel"]
    else:
        bev_agent_channel = policy_model_run.config["bev_agent_channel"]

    if "bev_vehicle_channel" in config["tester"]:
        bev_vehicle_channel = config["tester"]["bev_vehicle_channel"]
    else:
        bev_vehicle_channel = policy_model_run.config["bev_vehicle_channel"]

    if "bev_selected_channels" in config["tester"]:
        bev_selected_channels = config["tester"]["bev_selected_channels"]
    else:
        bev_selected_channels = policy_model_run.config["bev_selected_channels"]

    if "bev_calculate_offroad" in config["tester"]:
        bev_calculate_offroad = config["tester"]["bev_calculate_offroad"]
    else:
        bev_calculate_offroad = policy_model_run.config["bev_calculate_offroad"]

    # ---------------------------------------------------------------------------- #
    #                                    TESTER                                    #
    # ---------------------------------------------------------------------------- #
    tester_class = tester_factory(config)
    tester = tester_class(
        environment=environment,
        ego_forward_model=ego_forward_model,
        world_forward_model=world_forward_model,
        policy_model=policy_model,
        cost=cost,
        device=device,
        skip_frames=config["tester"]["skip_frames"],
        repeat_frames=config["tester"]["repeat_frames"],
        log_video=config["tester"]["log_video"],
        log_video_scale=config["tester"]["log_video_scale"],
        num_time_step_previous=policy_model_run.config["num_time_step_previous"],
        num_time_step_future=config["tester"]["rollout_length"],
        binary_occupancy=policy_model_run.config["training"]["binary_occupancy"][
            "enable"
        ],
        binary_occupancy_threshold=policy_model_run.config["training"][
            "binary_occupancy"
        ]["threshold"],
        use_world_forward_model_encoder_output_as_world_state=policy_model_run.config[
            "training"
        ]["use_world_forward_model_encoder_output_as_world_state"],
        bev_agent_channel=bev_agent_channel,
        bev_vehicle_channel=bev_vehicle_channel,
        bev_selected_channels=bev_selected_channels,
        bev_calculate_offroad=bev_calculate_offroad,
    )

    try:
        logger.info("Starting the tester")

        tester.test(run)

        logger.info("Tester finished")

    except Exception as e:
        logger.exception("Tester failed!", exc_info=e)

        logger.info("Closing the environment")

        run.log(
            {
                "INTERRUPTED": True,
                "SUCCESSFUL": False,
                "COLLISION": False,
            }
        )

        run.finish()
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/policy_model/testing/config_simple_bev.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = parse_yml(args.config_path)

    config["config_path"] = args.config_path

    main(config)
