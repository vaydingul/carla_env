import argparse
import logging
import os
import sys
import time
from pathlib import Path
import tqdm
import cv2
import numpy as np

from carla_env import carla_env_bev_data_collect
from carla_env.writer.writer import InstanceWriter, InstanceWriterType
from utils.train_utils import seed_everything
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):
    seed_everything(333)
    c = carla_env_bev_data_collect.CarlaEnvironment(
        config={
            "render": False, "save": False, "save_video": False,
            "tasks": [
                {
                    "world": "Town02", "num_vehicles": [60, 80]},
            ],
            "max_steps": 1000,
            "random": False,
            "fixed_delta_seconds": 1 / 10})

    for k in tqdm.tqdm(range(config.num_episodes)):
        
        # Create the data writer
        data_save_path_ = Path(config.data_save_path) / f"episode_{k}"
        os.makedirs(data_save_path_, exist_ok=True)

        writer = InstanceWriter(data_save_path_)

        # Add the keys to the writer
        writer.add_key(
            key="rgb_front",
            value="rgb_front",
            type=InstanceWriterType.RGB_IMAGE)
        # writer.add_key(
        #     key="rgb_left",
        #     value="rgb_left",
        #     type=InstanceWriterType.RGB_IMAGE)
        # writer.add_key(
        #     key="rgb_right",
        #     value="rgb_right",
        #     type=InstanceWriterType.RGB_IMAGE)
        writer.add_key(
            key="bev_world",
            value="bev_world",
            type=InstanceWriterType.BEV_IMAGE)
        writer.add_key(
            key="bev_ego",
            value="bev_ego",
            type=InstanceWriterType.BEV_IMAGE)
        writer.add_key(key="ego", value="ego", type=InstanceWriterType.JSON)
        writer.add_key(
            key="navigation",
            value="navigation",
            type=InstanceWriterType.JSON)
        writer.add_key(key="occ", value="occ", type=InstanceWriterType.JSON)

        while not c.is_done:

            c.step()

            data_ = c.get_data()

            c.render(bev_list=[data_["bev_world"], data_["bev_ego"]])

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
        default="./data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33",
        help="Path to save the data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to collect data from")
    config = parser.parse_args()

    main(config)
