from carla_env.dataset.instance import InstanceDataset
import os
import logging
import torch
import argparse
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    dataset = InstanceDataset(
        data_path=config.dataset_path,
        sequence_length=config.num_time_step_previous +
        config.num_time_step_future,
        dilation=config.dataset_dilation,
        read_keys=["bev_world", "ego"],
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
        bev_calculate_offroad=False)

    logger.info(f"Train dataset size: {len(dataset)}")

    if os.path.exists(f"{config.dataset_path}/weights.pt"):
        logger.info("Loading weights from file")
        weights = torch.load(f"{config.dataset_path}/weights.pt")
    else:
        logger.info("Calculating weights")
        weights = torch.Tensor(
            [dataset.__getweight__(k) for k in tqdm(range(
                len(dataset)))])
        torch.save(weights, f"{config.dataset_path}/weights.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev/",
        help="Path to the dataset")
    parser.add_argument(
        "--num_time_step_previous",
        type=int,
        default=5,
        help="Number of time steps in the past")
    parser.add_argument(
        "--num_time_step_future",
        type=int,
        default=10,
        help="Number of time steps in the future")
    parser.add_argument(
        "--dataset_dilation",
        type=int,
        default=2,
        help="Dilation of the dataset")
    config = parser.parse_args()

    main(config)
