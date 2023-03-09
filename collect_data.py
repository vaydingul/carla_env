import argparse
import logging
import os
import sys
import time
from pathlib import Path
import tqdm
import cv2
import numpy as np

from utils.path_utils import check_latest_episode
from utils.factory import *
from utils.wandb_utils import create_wandb_run
from utils.config_utils import parse_yml
from utils.log_utils import get_logger, configure_logger, pretty_print_config


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    pretty_print_config(logger, config)

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

    env_class = environment_factory(config)
    env = env_class(config=config["environment"])

    for k in tqdm.tqdm(range(config["num_episodes"])):

        latest_episode = check_latest_episode(config["data_save_path"])
        # Create the data writer
        data_save_path_ = Path(config["data_save_path"]) / f"episode_{latest_episode}"
        os.makedirs(data_save_path_, exist_ok=True)

        # ---------------------------------------------------------------------------- #
        #                                    WRITER                                    #
        # ---------------------------------------------------------------------------- #
        writer_class = writer_factory(config)

        writer = writer_class(data_save_path_)

        writer_key_list = writer_key_factory(config)

        for writer_key in writer_key_list:
            # Add the keys to the writer
            writer.add_key(**writer_key)

        while not env.is_done:

            tic = time.time()

            env.step()

            toc = time.time()

            data_ = env.get_data()

            env.render()

            writer.write(env.get_counter(), data_)

            run.log({"sim/step_time": 1 / (toc - tic)})
            run.log({"sim/num_frames": env.get_counter()})
            time.sleep(0.1)

        env.reset()

    env.close()

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/collect_data/driving/config1.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["config_path"] = args.config_path

    main(config)
