import os
import numpy as np
import cv2
from carla_env.bev import BirdViewProducer

episode_path = "/home/volkan/Documents/Codes/carla_env/data/driving_model_data_20Hz_action_repeat_1/test/episode_6/"
os.makedirs(os.path.join(episode_path, "bev_world_img"), exist_ok=True)
for bev_npz_file in os.listdir(os.path.join(episode_path, "bev_world")):
    if bev_npz_file.endswith("npz"):
        data = np.load(
            os.path.join(os.path.join(episode_path, "bev_world"), bev_npz_file)
        )
        bev = data["bev"]
        bev = bev.transpose(1, 2, 0)
        rgb = BirdViewProducer.as_rgb(bev)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                os.path.join(episode_path, "bev_world_img"),
                bev_npz_file.replace("npz", "png"),
            )
        )
for bev_npz_file in os.listdir(os.path.join(episode_path, "bev_world")):
    if bev_npz_file.endswith("npz"):
        data = np.load(
            os.path.join(os.path.join(episode_path, "bev_world"), bev_npz_file)
        )
        bev = data["bev"]
        # bev = bev.transpose(1,2,0)
        rgb = BirdViewProducer.as_rgb(bev)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                os.path.join(episode_path, "bev_world_img"),
                bev_npz_file.replace("npz", "png"),
            )
        )
for bev_npz_file in os.listdir(os.path.join(episode_path, "bev_world")):
    if bev_npz_file.endswith("npz"):
        data = np.load(
            os.path.join(os.path.join(episode_path, "bev_world"), bev_npz_file)
        )
        bev = data["bev"]
        # bev = bev.transpose(1,2,0)
        rgb = BirdViewProducer.as_rgb(bev)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                os.path.join(episode_path, "bev_world_img"),
                bev_npz_file.replace("npz", "png"),
            ),
            rgb,
        )
