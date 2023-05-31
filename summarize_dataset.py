import logging
import argparse
import numpy as np
from tqdm import tqdm
import os
from utilities.config_utils import parse_yml
from utilities.factory import *
from utilities.log_utils import configure_logger, get_logger


def main(config):
    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(config)

    # ---------------------------------------------------------------------------- #
    #                         TRAIN AND VALIDATION DATASETS                        #
    # ---------------------------------------------------------------------------- #
    dataset = dataset_class(config["dataset"]["config"])
    logger.info(f"Dataset size: {len(dataset)}")

    if "bev_world" in dataset.read_keys:
        bev_information = np.zeros(
            (
                len(dataset),
                len(dataset.bev_selected_channels) + dataset.bev_calculate_offroad,
            )
        )

    if "ego" in dataset.read_keys:
        ego_rotation_information = np.zeros((len(dataset),))
        ego_speed_information = np.zeros((len(dataset),))
        ego_control_throttle_information = np.zeros((len(dataset),))
        ego_control_steer_information = np.zeros((len(dataset),))
        ego_control_brake_information = np.zeros((len(dataset),))

    if "navigation" in dataset.read_keys:
        navigation_command_information = np.zeros((len(dataset),))

    if "navigation_downsampled" in dataset.read_keys:
        navigation_downsampled_command_information = np.zeros((len(dataset),))

    if "occ" in dataset.read_keys:
        occ_information = np.zeros((len(dataset), 8))

    navigational_command_list = np.zeros((6,))
    for ix, k in enumerate(tqdm(range(0, len(dataset)))):
        # Get data
        data = dataset[k]

        if "bev_world" in dataset.read_keys:
            bev_information[ix] = data["bev_world"]["bev"].numpy()[0].sum(axis=(1, 2))

        if "ego" in dataset.read_keys:
            ego_rotation_information[ix] = data["ego"]["rotation_array"].numpy()[0][2]
            ego_speed_information[ix] = np.linalg.norm(
                data["ego"]["velocity_array"].numpy()[0]
            )
            ego_control_throttle_information[ix] = data["ego"]["control_array"].numpy()[
                0
            ][0]
            ego_control_steer_information[ix] = data["ego"]["control_array"].numpy()[0][
                1
            ]
            ego_control_brake_information[ix] = data["ego"]["control_array"].numpy()[0][
                2
            ]

        if "navigation" in dataset.read_keys:
            navigation_command_information[ix] = (
                data["navigation"]["command"].numpy()[0] - 1
            )

        if "navigation_downsampled" in dataset.read_keys:
            navigation_downsampled_command_information[ix] = (
                data["navigation_downsampled"]["command"].numpy()[0] - 1
            )

        if "occ" in dataset.read_keys:
            occ_information[ix] = data["occ"]["occupancy"].numpy()[0]

    dataset_information = {}

    if "bev_world" in dataset.read_keys:
        for k in range(len(dataset.bev_selected_channels)):
            histogram, bins = np.histogram(bev_information[:, k], bins=100)
            dataset_information[f"bev_world_{k}"] = {
                "histogram": histogram,
                "bins": bins,
            }

    if "ego" in dataset.read_keys:
        histogram, bins = np.histogram(ego_rotation_information, bins=100)
        dataset_information["ego_rotation"] = {"histogram": histogram, "bins": bins}

        histogram, bins = np.histogram(ego_speed_information, bins=100)
        dataset_information["ego_speed"] = {"histogram": histogram, "bins": bins}

        histogram, bins = np.histogram(ego_control_throttle_information, bins=100)
        dataset_information["ego_control_throttle"] = {
            "histogram": histogram,
            "bins": bins,
        }

        histogram, bins = np.histogram(ego_control_steer_information, bins=100)
        dataset_information["ego_control_steer"] = {
            "histogram": histogram,
            "bins": bins,
        }

        histogram, bins = np.histogram(ego_control_brake_information, bins=100)
        dataset_information["ego_control_brake"] = {
            "histogram": histogram,
            "bins": bins,
        }

    if "navigation" in dataset.read_keys:
        histogram, bins = np.histogram(navigation_command_information, bins=100)
        dataset_information["navigation_command"] = {
            "histogram": histogram,
            "bins": bins,
        }

    if "navigation_downsampled" in dataset.read_keys:
        histogram, bins = np.histogram(
            navigation_downsampled_command_information, bins=100
        )
        dataset_information["navigation_downsampled_command"] = {
            "histogram": histogram,
            "bins": bins,
        }

    if "occ" in dataset.read_keys:
        for k in range(8):
            histogram, bins = np.histogram(occ_information[:, k], bins=100)
            dataset_information[f"occ_{k}"] = {"histogram": histogram, "bins": bins}

    # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    #                                 SAVE DATASET                                 #
    # ---------------------------------------------------------------------------- #
    save_path = os.path.join(
        config["dataset"]["config"]["data_path"], f"{'_'.join(dataset.read_keys)}.npz"
    )
    np.savez(save_path, dataset_information)


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
