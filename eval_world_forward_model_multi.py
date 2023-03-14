import argparse
from datetime import datetime
from pathlib import Path
import logging
import yaml
import os


from utils.config_utils import parse_yml
from utils.factory import *
from utils.log_utils import get_logger, configure_logger, pretty_print_config


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #

    logger = get_logger(__name__)
    configure_logger(__name__, log_level=logging.INFO)
    pretty_print_config(logger, config)

    # ---------------------------------------------------------------------------- #

    for wandb_dict in config["wandbs"]:

        for checkpoint_number in wandb_dict["checkpoint_numbers"]:

            wandb_link = wandb_dict["link"]

            logger.info(f"Starting evaluation for {wandb_link} - {checkpoint_number}")

            # ---------------------------------------------------------------------------- #

            config_specific = parse_yml(config["config_path"])

            # ---------------------------------------------------------------------------- #
            wandb_id = wandb_link.split("/")[-1]
            save_path = Path(config_specific["save_path"])
            save_path = save_path / f"{wandb_id}" / f"{checkpoint_number}"
            save_path.mkdir(parents=True, exist_ok=True)

            config_specific["save_path"] = str(save_path)

            # ---------------------------------------------------------------------------- #

            config_specific["wandb"]["link"] = wandb_link
            config_specific["wandb"]["checkpoint_number"] = checkpoint_number

            # ---------------------------------------------------------------------------- #

            file_dir = f"/tmp/config_{wandb_id}_{checkpoint_number}.yml"

            # Open a temporary file in /tmp dir
            with open(file_dir, "w") as f:
                yaml.dump(config_specific, f)

            # ---------------------------------------------------------------------------- #

            os.system(f"python3 eval_world_forward_model.py --config_path {file_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/world_forward_model/evaluation_multi/config.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)

    main(config)
