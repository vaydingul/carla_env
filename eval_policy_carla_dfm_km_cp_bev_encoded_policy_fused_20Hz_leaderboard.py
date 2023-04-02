from carla_env.carla_env_leaderboard_headless import CarlaEnvironment
from carla_env.models.dfm_km_cp import DecoupledForwardModelKinematicsCoupledPolicy
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy_fused import Policy
from carla_env.cost.masked_cost_batched_policy_bev import Cost
from carla_env.bev import BirdViewProducer
import torch
import time
import logging
import wandb
import numpy as np
import math
import argparse
import cv2
from collections import deque
from utils.config_utils import parse_yml
from utils.kinematic_utils import acceleration_to_throttle_brake

from utils.train_utils import seed_everything, get_device
from leaderboard.leaderboard_evaluator_local import main as leaderboard_evaluator_main

logging.basicConfig(level=logging.INFO)


def main(config):

    # config.seed = int(np.random.rand(1) * 1000)
    seed_everything(config["seed"])

    device = get_device()

    environment = CarlaEnvironment(config={})

    leaderboard_evaluator_main(config["leaderboard"], device, environment)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path_agent", type=str, default="configs/config_agent.yml"
    )
    parser.add_argument(
        "--config_path_leaderboard", type=str, default="configs/config_leaderboard.yml"
    )

    args = parser.parse_args()

    config_leaderboard = parse_yml(args.config_path_leaderboard)
    config_leaderboard["config_path_leaderboard"] = args.config_path_leaderboard
    config_leaderboard["config_path_agent"] = args.config_path_agent
    config_leaderboard["leaderboard"]["agent_config"] = args.config_path_agent

    main(config_leaderboard)
