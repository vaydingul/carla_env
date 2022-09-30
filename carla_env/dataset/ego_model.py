import torch
from torch.utils.data import Dataset
import os
import numpy as np


class EgoModelDataset(Dataset):

    def __init__(self, data_path, rollout_length=10, dataset_crop = 0):

        self.location = []
        self.rotation = []
        self.velocity = []
        self.acceleration = []
        self.control = []
        self.elapsed_time = []
        self.count_array = []

        self.rollout_length = rollout_length

        for file in os.listdir(data_path):
            if file.endswith(".npz"):
                data_ = np.load(os.path.join(data_path, file))
                self.location.append(data_["vehicle_location"][dataset_crop:])
                self.rotation.append(data_["vehicle_rotation"][dataset_crop:])
                self.velocity.append(data_["vehicle_velocity"][dataset_crop:])
                self.acceleration.append(data_["vehicle_acceleration"][dataset_crop:])
                self.control.append(data_["vehicle_control"][dataset_crop:])
                self.elapsed_time.append(data_["elapsed_time"][dataset_crop:])
                self.count_array.append(self.location[-1].shape[0]
                                        + (self.count_array[-1] if len(self.count_array) > 0 else 0))

        self.location = torch.Tensor(np.concatenate(self.location, axis=0))
        self.rotation = torch.Tensor(np.concatenate(self.rotation, axis=0))
        self.velocity = torch.Tensor(np.concatenate(self.velocity, axis=0))
        self.acceleration = torch.Tensor(
            np.concatenate(self.acceleration, axis=0))
        self.control = torch.Tensor(np.concatenate(self.control, axis=0))
        self.elapsed_time = torch.Tensor(
            np.concatenate(self.elapsed_time, axis=0))
        self.count_array = np.array(self.count_array)

    def __len__(self) -> int:
        return self.elapsed_time.shape[0] - self.rollout_length

    def __getitem__(self, index):

        I, = np.nonzero(np.logical_and((((index + self.rollout_length) - self.count_array)
                                        >= 0), (((index + self.rollout_length) - self.count_array) < self.rollout_length)))

        if I.size != 0:
            index = self.count_array[I[-1]]

        return self.location[index: index + self.rollout_length, :], self.rotation[index: index + self.rollout_length, :], self.velocity[index: index + self.rollout_length, :], self.acceleration[index: index + self.rollout_length, :], self.control[index: index + self.rollout_length, :], self.elapsed_time[index: index + self.rollout_length]



class EgoModelDatasetV2(Dataset):

    def __init__(self, data_path, rollout_length=10, dataset_crop = 0):

        self.location = []
        self.rotation = []
        self.velocity = []
        self.acceleration = []
        self.control = []
        self.elapsed_time = []
        self.count_array = []

        self.rollout_length = rollout_length

        for file in os.listdir(data_path):
            if file.endswith(".npz"):
                data_ = np.load(os.path.join(data_path, file))
                self.location.append(data_["vehicle_location"][dataset_crop:])
                self.rotation.append(data_["vehicle_rotation"][dataset_crop:])
                self.velocity.append(data_["vehicle_velocity"][dataset_crop:])
                self.acceleration.append(data_["vehicle_acceleration"][dataset_crop:])
                self.control.append(data_["vehicle_control"][dataset_crop:])
                self.elapsed_time.append(data_["elapsed_time"][dataset_crop:])
                self.count_array.append(self.location[-1].shape[0]
                                        + (self.count_array[-1] if len(self.count_array) > 0 else 0))

        self.location = torch.Tensor(np.concatenate(self.location, axis=0))
        self.rotation = torch.Tensor(np.concatenate(self.rotation, axis=0))
        self.velocity = torch.Tensor(np.concatenate(self.velocity, axis=0))
        self.acceleration = torch.Tensor(
            np.concatenate(self.acceleration, axis=0))
        self.control = torch.Tensor(np.concatenate(self.control, axis=0))
        self.elapsed_time = torch.Tensor(
            np.concatenate(self.elapsed_time, axis=0))
        self.count_array = np.array(self.count_array)

    def __len__(self) -> int:
        return self.elapsed_time.shape[0] - self.rollout_length

    def __getitem__(self, index):

        I, = np.nonzero(np.logical_and((((index + self.rollout_length) - self.count_array)
                                        >= 0), (((index + self.rollout_length) - self.count_array) < self.rollout_length)))

        if I.size != 0:
            index = self.count_array[I[-1]]

        throttle_ = self.control[index: index + self.rollout_length, 0]
        steer_ = self.control[index: index + self.rollout_length, 1]
        brake_ = self.control[index: index + self.rollout_length, 2]

        acceleration_ = ((throttle_ > 0) * throttle_) + ((brake_ > 0) * -brake_)

        control_ = torch.stack((acceleration_, steer_), dim=1)

        return self.location[index: index + self.rollout_length, :], self.rotation[index: index + self.rollout_length, :], self.velocity[index: index + self.rollout_length, :], self.acceleration[index: index + self.rollout_length, :], control_, self.elapsed_time[index: index + self.rollout_length]


