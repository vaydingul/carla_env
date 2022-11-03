import argparse
import logging
import os
import sys
from this import d
import time
from pathlib import Path

import cv2
import numpy as np

from carla_env import carla_env_bev_data_collect
from carla_env.writer.writer import InstanceWriter, InstanceWriterType
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):
    c = carla_env_bev_data_collect.CarlaEnvironment(
        config={
            "render": False, "save": False, "save_video": False,
            "tasks": [
                {
                    "world": "Town02", "num_vehicles": 100},
            ],
            "max_steps": 1000})

    for k in range(config.num_episodes):

        # Create the data writer
        data_save_path_ = Path(config.data_save_path) / f"episode_{k + 6}"
        os.makedirs(data_save_path_, exist_ok=True)

        writer = InstanceWriter(data_save_path_)

        # Add the keys to the writer
        writer.add_key(
            key="rgb_front",
            value="rgb_front",
            type=InstanceWriterType.RGB_IMAGE)
        writer.add_key(
            key="rgb_left",
            value="rgb_left",
            type=InstanceWriterType.RGB_IMAGE)
        writer.add_key(
            key="rgb_right",
            value="rgb_right",
            type=InstanceWriterType.RGB_IMAGE)
        writer.add_key(
            key="bev",
            value="bev",
            type=InstanceWriterType.BEV_IMAGE)
        writer.add_key(key="ego", value="ego", type=InstanceWriterType.JSON)

        while not c.is_done:

            c.step()

            data_ = c.get_data()

            c.render(bev=data_["bev"])

            writer.write(c.get_counter(), data_)
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
        default=1,
        help="Number of episodes to collect data from")
    config = parser.parse_args()

    main(config)
