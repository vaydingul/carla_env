import argparse
import logging
import time
import numpy as np
import carla
from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.config_utils import parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config


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
    #                                    NOISER                                    #
    # ---------------------------------------------------------------------------- #

    noiser_class = noiser_factory(config)
    config["environment"].update({"noiser": noiser_class})

    # ---------------------------------------------------------------------------- #
    #                                  ENVIRONMENT                                 #
    # ---------------------------------------------------------------------------- #

    env_class = environment_factory(config)
    env = env_class(config=config["environment"])

    
    while not env.is_done:

        tic = time.time()

        env.step()

        toc = time.time()

        data_ = env.get_data()

        env.render()

        run.log({"sim/step_time": 1 / (toc - tic)})
        run.log({"sim/num_frames": env.get_counter()})
        time.sleep(0.1)

    env.close()

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/play/config.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["config_path"] = args.config_path

    main(config)
