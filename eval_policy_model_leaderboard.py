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

from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.config_utils import parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    # seed_everything(config["seed"])
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
    #                                    EVALUATOR                                 #
    # ---------------------------------------------------------------------------- #
    evaluator_class = evaluator_factory(config)
    evaluator = evaluator_class(
        environment=environment,
        leaderboard=config["leaderboard"],
        device=device,
    )

    evaluator.evaluate(run)
    run.finish()
    environment.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path_leaderboard",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/policy_model/evaluation/config_leaderboard.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--config_path_agent",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/policy_model/evaluation/config_agent.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config_leaderboard = parse_yml(args.config_path_leaderboard)

    config_leaderboard["config_path_leaderboard"] = args.config_path_leaderboard
    config_leaderboard["config_path_agent"] = args.config_path_agent
    config_leaderboard["leaderboard"]["agent_config"] = args.config_path_agent
    main(config_leaderboard)
