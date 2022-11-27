import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from carla_env import carla_env_playground
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):
    c = carla_env_playground.CarlaEnvironment(
        config={"render": True, "save": False, "save_video": False,
                "max_steps": 10000})

    for k in range(config.num_episodes):

        while not c.is_done:

            c.step()

            data_ = c.get_data()

            c.render(bev=data_["bev"])

            time.sleep(0.1)

        c.reset()

    c.close()

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")
    parser.add_argument(
        "--data_save_path",
        type=str,
        default="./data/ground_truth_bev_model_test_data_2",
        help="Path to save the data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to collect data from")
    config = parser.parse_args()

    main(config)
