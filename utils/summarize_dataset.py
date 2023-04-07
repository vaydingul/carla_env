import logging
import argparse
import numpy as np

from utils.config_utils import parse_yml
from utils.factory import *


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s",
)


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(config)

    # ---------------------------------------------------------------------------- #
    #                         TRAIN AND VALIDATION DATASETS                        #
    # ---------------------------------------------------------------------------- #
    dataset = dataset_class(config["dataset"]["config"])
    logger.info(f"Dataset size: {len(dataset)}")

    navigational_command_list = np.zeros((6,))
    for i in range(len(dataset)):
        # Get data
        data = dataset[i]
        navigational_command = data["navigation"]["command"]
        navigational_command_list[int(navigational_command) - 1] += 1

    print(navigational_command_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/dataset_summary/config.yml"
    )

    config = parser.parse_args()
    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["config_path"] = args.config_path

    main(config)
