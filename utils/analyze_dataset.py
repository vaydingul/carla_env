from carla_env.dataset.instance import InstanceDataset
import logging
import argparse
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s",
)


def main(config):
    bev_selected_channels = [3, 4, 6]
    # Create dataset and its loader
    dataset_path = config.dataset_path
    dataset = InstanceDataset(
        data_path=dataset_path,
        sequence_length=1,
        read_keys=["bev_world", "rgb_front", "ego"],
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=bev_selected_channels,
        bev_calculate_offroad=False,
    )

    speed = np.empty((len(dataset),))
    yaw = np.empty((len(dataset),))
    bev_channels = np.empty((len(dataset), len(bev_selected_channels)))

    logger.info(f"Dataset size: {len(dataset)}")

    for (ix, i) in enumerate(tqdm(range(0, len(dataset)))):
        # Get data
        data = dataset[i]
        bev = data["bev_world"]["bev"][0].numpy()
        ego = data["ego"]
        speed[ix] = np.linalg.norm(ego["velocity_array"][0])
        yaw[ix] = ego["rotation_array"][0][2]
        bev_channels[ix] = np.sum(bev, axis=(1, 2))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(speed, bins=50)
    axs[0].set_title("Speed Distribution")
    axs[0].set_xlabel("Speed (m/s)")
    axs[0].set_ylabel("Count")

    axs[1].hist(
        bev_channels, bins=50, label=["Channel {}".format(i) for i in range(12)]
    )
    axs[1].set_title("BEV Channel Distribution")
    axs[1].set_xlabel("Channel Value")
    axs[1].set_ylabel("Count")
    axs[1].legend()

    axs[2].hist(yaw, bins=50)
    axs[2].set_title("Yaw Distribution")
    axs[2].set_xlabel("Yaw (deg)")
    axs[2].set_ylabel("Count")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev/",
    )
    config = parser.parse_args()

    main(config)
