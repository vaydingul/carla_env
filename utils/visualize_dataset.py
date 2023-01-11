
from carla_env.dataset.instance import InstanceDataset
import torch
from torch.utils.data import DataLoader, Subset
import logging
import argparse
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from carla_env.bev import BirdViewProducer
import os
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def bev_to_rgb(bev):

    # Transpose the bev representation
    bev = bev.transpose(1, 2, 0)

    rgb_image = BirdViewProducer.as_rgb(bev)

    return rgb_image


def main(config):

    # Create dataset and its loader
    data_path_test = config.data_path_test
    world_model_dataset_test = InstanceDataset(
        data_path=data_path_test,
        sequence_length=1,
        read_keys=["bev_world", "rgb_front", "ego"],
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=range(12),
        bev_calculate_offroad=False)
    
    world_model_dataset_test = Subset(world_model_dataset_test, range(1000))

    logger.info(f"Test dataset size: {len(world_model_dataset_test)}")

    rgb_size = (
        world_model_dataset_test[0]["rgb_front"].shape[2],
        world_model_dataset_test[0]["rgb_front"].shape[3])
    bev_size = (
        world_model_dataset_test[0]["bev_world"]["bev"].shape[2],
        world_model_dataset_test[0]["bev_world"]["bev"].shape[3])
    # Create video writer
    canvas = np.zeros(
        (rgb_size[0] +
         bev_size[0] * 3 +
         100,
         rgb_size[1],
         3),
        dtype=np.uint8)

    # Create the folder for save path
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fourcc = VideoWriter_fourcc(*'mp4v')
    video = VideoWriter(f"{config.save_path}/video.mp4",
                        fourcc, 10.0, (canvas.shape[1], canvas.shape[0]))

    for i in range(len(world_model_dataset_test)):
        # Get data
        data = world_model_dataset_test[i]
        bev = data["bev_world"]["bev"][0].numpy()
        rgb_front = data["rgb_front"][0].numpy().transpose(1, 2, 0)
        ego = data["ego"]

        # Create canvas
        canvas = np.zeros_like(canvas)
        canvas[:rgb_size[0], :rgb_size[1], :] = cv2.cvtColor(
            rgb_front, cv2.COLOR_RGB2BGR)
        canvas[rgb_size[0] + 100:rgb_size[0] + 100 +  3 *bev_size[0],
               :bev_size[1] * 3] = cv2.resize(cv2.cvtColor(bev_to_rgb(bev), cv2.COLOR_RGB2BGR), (0,0), fx=3, fy=3)

        # Put text on the top-middle of the images and bev
        cv2.putText(canvas, "Front", (rgb_size[1] // 2 - 50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        
        cv2.putText(
            canvas,
            "Bird's Eye View",
            (bev_size[1] // 2 - 50,
             rgb_size[0] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,
             255,
             255),
            2)

        offset_x = bev_size[1] + 100
        offset_y = rgb_size[0] + 100

        # for key, value in ego.items():

        #     cv2.putText(
        #         canvas,
        #         f"{key}: {value[0].numpy()}",
        #         (offset_x,
        #          offset_y),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1.0,
        #         (255,
        #          255,
        #          0),
        #         1)
        #     offset_y += 40

        # Write to video
        video.write(canvas)

        # if i == 100:
        #     break
    video.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_test", type=str,
                        default="data/ground_truth_bev_model_test_data/")
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures/world_model_dataset_visualization/")

    config = parser.parse_args()

    main(config)
