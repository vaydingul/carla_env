import argparse
import logging
import time
import numpy as np

from utils.factory import *
from utils.wandb_utils import create_wandb_run
from utils.config_utils import parse_yml
from utils.log_utils import get_logger, configure_logger, pretty_print_config


def callback(hero_actor, world_snapshot):
    logger = get_logger(__name__)
    logger.info("Callback is called!")


    if np.random.rand() < 0.05:
        
        # Get control from hero actor
        hero_actor.set_autopilot(False)
        logger.info(hero_actor.attributes)
        action = hero_actor.get_control()
        logger.info(f"Throttle: {action.throttle}")
        logger.info(f"Steer: {action.steer}")
        logger.info(f"Brake: {action.brake}")
        action.throttle = 0.5
        action.steer = 0.0
        action.brake = 0.0
        hero_actor.apply_control(action)

    else:

        hero_actor.set_autopilot(False)


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

    env.get_world().on_tick(
        lambda world_snapshot: callback(env.get_hero_actor(), world_snapshot)
    )
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
