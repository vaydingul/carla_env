from ast import arg
from carla_env import carla_env_bev_data_collect

import time
import logging
import cv2
import os
import numpy as np
import argparse
import sys
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):

    for k in range(config.num_episodes):

        c = carla_env_bev_data_collect.CarlaEnvironment(config={
            "render": False,
            "save": False,
            "save_video": False,
            "worlds": ["Town01", "Town02"]})

        t_init = time.time()

        while not c.is_done:

            c.step()
            # TODO: Write data to a file here

        c.close()

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")
    parser.add_argument(
        "--data_save_path",
        type=str,
        default="./data/kinematic_model_data_train_3",
        help="Path to save the data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect data from")
    config = parser.parse_args()

    main(config)
