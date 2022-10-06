import numpy as np
import matplotlib.pyplot as plt
from carla_env.models.dynamic.vehicle import KinematicBicycleModel, KinematicBicycleModelV2
from carla_env.models.dynamic.vehicle_WoR import EgoModel
#from ego_model import EgoModel
import logging
import os
import pathlib
import torch
import argparse
from utils.plot_utils import plot_result_eval
from utils.kinematic_utils import throttle_brake_to_acceleration
import time

logger = logging.getLogger(__name__)


def evaluate(fname, data, model, config):

    vehicle_location = torch.Tensor(data["vehicle_location"])
    vehicle_rotation = torch.Tensor(data["vehicle_rotation"])
    vehicle_velocity = torch.Tensor(data["vehicle_velocity"])
    vehicle_control = torch.Tensor(data["vehicle_control"])
    elapsed_time = torch.Tensor(data["elapsed_time"])

    logger.info(f"Total Elapsed Time: {elapsed_time[-1]}")
    logger.info(f"Number of Steps: {elapsed_time.shape[0]}")

    location_predicted = []
    yaw_predicted = []
    speed_predicted = []

    for k in range(0, elapsed_time.shape[0] - 1):

        if k % 10 == 0:

            location = vehicle_location[k, :2]
            yaw = vehicle_rotation[k, 1:2]
            speed = torch.norm(vehicle_velocity[k, :], dim=-1, keepdim=True)

            location_predicted.append(location)
            yaw_predicted.append(yaw)
            speed_predicted.append(speed)

            continue

        action = vehicle_control[k, :]

        if config.kinematic_model == "v2":
            acceleration = throttle_brake_to_acceleration(
                action[..., 0], action[..., 2])
            action = torch.stack([acceleration, action[..., 1]], dim=-1)
            location, yaw, speed = model(location, yaw, speed, action)
        else:
            location, yaw, speed = model(location, yaw, speed, action)

        location_predicted.append(location)
        yaw_predicted.append(yaw)
        speed_predicted.append(speed)

    location_predicted = torch.stack(
        location_predicted, dim=0).detach().numpy()
    yaw_predicted = torch.stack(yaw_predicted, dim=0).detach().numpy()
    speed_predicted = torch.stack(speed_predicted, dim=0).detach().numpy()

    vehicle_location = vehicle_location.numpy()
    vehicle_rotation = vehicle_rotation.numpy()

    location_loss = np.mean(
        np.abs(vehicle_location[:-1, :2] - location_predicted))
    rotation_loss = np.mean(
        np.abs(np.cos(vehicle_rotation[:-1, 1:2]) - np.cos(yaw_predicted)))
    rotation_loss += np.mean(
        np.abs(np.sin(vehicle_rotation[:-1, 1:2]) - np.sin(yaw_predicted)))
    rotation_loss_raw = np.rad2deg(
        np.mean(np.abs(vehicle_rotation[:-1, 1:2] - yaw_predicted)))

    savedir = pathlib.Path(f"{config.save_dir}/{fname}")
    os.makedirs(savedir, exist_ok=True)

    logger.info(savedir)
    logger.info(
        f"Location MAE: {np.mean(np.abs(vehicle_location[:-1, :2] - location_predicted))}")
    logger.info(
        f"Orientation Cos MAE: {np.mean(np.abs(np.cos(vehicle_rotation[:-1, 1:2]) - np.cos(yaw_predicted)))}")
    logger.info(
        f"Orientation Sin MAE: {np.mean(np.abs(np.sin(vehicle_rotation[:-1, 1:2]) - np.sin(yaw_predicted)))}")

    if config.plot_local:
        plot_result_eval(
            vehicle_location,
            vehicle_rotation,
            vehicle_velocity,
            vehicle_control,
            elapsed_time,
            location_predicted,
            yaw_predicted,
            speed_predicted,
            savedir)
        time.sleep(1)

    return location_loss, rotation_loss, rotation_loss_raw


def main(config):
    folder_name = config.evaluation_data_folder

    if config.kinematic_model == "v1":
        model = KinematicBicycleModel(dt=1 / 20)
        model.state_dict = torch.load(config.model_path)
        model.eval()

    elif config.kinematic_model == "v2":
        model = KinematicBicycleModelV2(dt=1 / 20)
        model.state_dict = torch.load(config.model_path)
        model.eval()
    else:
        model = EgoModel(dt=1 / 20)
        model.load_state_dict(torch.load("pretrained_models/WoR/ego_model.th"))
        model.eval()

    config.save_dir = f"figures/ego-forward-model-evaluation/{config.save_dir}/"
    os.makedirs(config.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config.save_dir}/eval_ego_forward_model.log",
        filemode="w")

    location_loss_list = []
    rotation_loss_list = []
    rotation_loss_raw_list = []

    for file in os.listdir(folder_name):
        if file.endswith(".npz"):
            file_path = os.path.join(folder_name, file)
            data = np.load(file_path)

            location_loss, rotation_loss, rotation_loss_raw = evaluate(
                file, data, model, config)
            location_loss_list.append(location_loss)
            rotation_loss_list.append(rotation_loss)
            rotation_loss_raw_list.append(rotation_loss_raw)

    logger.info(f"Average Location Loss: {np.mean(location_loss_list)}")
    logger.info(f"Average Rotation Loss: {np.mean(rotation_loss_list)}")
    logger.info(
        f"Average Rotation Loss Raw: {np.mean(rotation_loss_raw_list)}")
    logger.info(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt")  # More data
    # parser.add_argument("--model_path", type=str,
    # default="pretrained_models/2022-09-28/03-24-39/ego_model_new.pt") # New
    # dataset
    parser.add_argument(
        "--evaluation_data_folder",
        type=str,
        default="./data/kinematic_model_data_test_3")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="train_new_model_test_new_dataset")
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--plot_local", type=bool, default=True)
    parser.add_argument("--kinematic_model", type=str, default="v2")
    config = parser.parse_args()

    main(config)
