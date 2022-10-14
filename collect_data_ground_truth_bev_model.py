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
    c = carla_env_bev_data_collect.CarlaEnvironment(config={
                "render": False,
                "save": False,
                "save_video": False,
                "worlds": ["Town01", "Town03", "Town04", "Town06"],
                "max_steps" : 1000})

    for k in range(config.num_episodes):

        # Create the data writer
        data_save_path_ = Path(config.data_save_path) / f"episode_{k}"
        os.makedirs(data_save_path_, exist_ok=True)

        writer = InstanceWriter(data_save_path_)

        # Add the keys to the writer
        writer.add_key("rgb_front", "rgb_front", InstanceWriterType.RGB_IMAGE)
        writer.add_key("rgb_left", "rgb_left", InstanceWriterType.RGB_IMAGE)
        writer.add_key("rgb_right", "rgb_right", InstanceWriterType.RGB_IMAGE)
        writer.add_key("bev", "bev", InstanceWriterType.BEV_IMAGE)
        writer.add_key("ego", "ego", InstanceWriterType.JSON)

        

        while not c.is_done:

            c.step()
            writer.write(c.get_counter(), c.get_data())
            # TODO: Write data to a file here

        # TODO: Do not close the environment, just reset it!
        c.reset()

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")
    parser.add_argument(
        "--data_save_path",
        type=str,
        default="./data/ground_truth_bev_model_train_data",
        help="Path to save the data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to collect data from")
    config = parser.parse_args()

    main(config)
