import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, default_collate
import os
import numpy as np
import cv2
import json
import logging
import scipy.stats

logger = logging.getLogger(__name__)

BEV_SCORE_LIMIT = []


class InstanceDataset(Dataset):
    """This dataset is very similar to InstanceWriter. It reads the data
    from different folders and appends to a Dict."""

    def __init__(
        self,
        config: dict,
    ):

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.count_array = []
        self.data = []

        for item in os.listdir(self.data_path):

            if item.startswith("episode"):

                episode_path = self.data_path / item

                key_lengths = np.array(
                    [
                        len(os.listdir(episode_path / read_key))
                        for read_key in self.read_keys
                    ]
                )
                if not np.all(key_lengths == key_lengths[0]):
                    print("All keys should include same amount of data!")
                    print(f"Skipping {episode_path}")

                key_ = self.read_keys[0]

                episode_key_path = episode_path / key_

                step_list = sorted(
                    [int(x.split(".")[0]) for x in os.listdir(episode_key_path)]
                )

                counter = 0
                for step in step_list:

                    self.data.append([episode_path, step])
                    counter += 1

                self.count_array.append(
                    counter + (self.count_array[-1] if len(self.count_array) > 0 else 0)
                )

        self.count_array.pop(-1)
        self.count_array = np.array(self.count_array)

    def build_from_config(self):

        self.data_path = Path(self.config["data_path"])
        self.sequence_length = self.config["sequence_length"]
        self.read_keys = self.config["read_keys"]
        self.dilation = self.config["dilation"]
        self.bev_agent_channel = self.config["bev_agent_channel"]
        self.bev_vehicle_channel = self.config["bev_vehicle_channel"]
        self.bev_selected_channels = self.config["bev_selected_channels"]
        self.bev_calculate_offroad = self.config["bev_calculate_offroad"]

    def __len__(self):
        return (
            len(self.data)
            - self.sequence_length
            - (self.sequence_length - 1) * (self.dilation - 1)
            + 1
        )

    def __getitem__(self, index):

        (I,) = np.nonzero(
            np.logical_and(
                (
                    (
                        (index + self.sequence_length * self.dilation - 1)
                        - self.count_array
                    )
                    >= 0
                ),
                (
                    (
                        (index + self.sequence_length * self.dilation - 1)
                        - self.count_array
                    )
                    <= self.sequence_length * self.dilation - 1
                ),
            )
        )

        if I.size != 0:
            index = self.count_array[I[-1]]

        data = {}

        # for read_key in self.read_keys:

        #     if read_key in ["bev", "bev_world", "bev_ego"]:

        #         load_function = self._load_bev

        #     if read_key in ["rgb_front", "rgb_left", "rgb_right"]:

        #         load_function = self._load_rgb

        #     if read_key in ["ego", "navigation", "occ", "navigation_downsampled"]:

        #         load_function = self._load_json

        #     data[read_key] = default_collate(
        #         [
        #             load_function(index + k, read_key)
        #             for k in range(
        #                 0, self.sequence_length * self.dilation, self.dilation
        #             )
        #         ]
        #     )

        for read_key in self.read_keys:

            if read_key in ["bev", "bev_world", "bev_ego"]:

                data_ = [
                    self._load_bev(index + k, read_key)
                    for k in range(
                        0, self.sequence_length * self.dilation, self.dilation
                    )
                ]

            if read_key in ["rgb_front", "rgb_left", "rgb_right"]:

                data_ = torch.stack(
                    [
                        self._load_rgb(index + k, read_key)
                        for k in range(
                            0, self.sequence_length * self.dilation, self.dilation
                        )
                    ],
                    dim=0,
                )

            if read_key in ["ego", "navigation", "occ", "navigation_downsampled"]:

                data_ = [
                    self._load_json(index + k, read_key)
                    for k in range(
                        0, self.sequence_length * self.dilation, self.dilation
                    )
                ]

            elem = data_[0]
            if isinstance(elem, dict):

                data_stacked = {}

                for key in elem.keys():
                    data_stacked[key] = torch.stack(
                        [data_[k][key] for k in range(len(data_))], dim=0
                    )

                data_ = data_stacked

            data[read_key] = data_

        return data

    def __getweight__(self, index):

        assert "bev_world" in self.read_keys, "bev_world should be in read_keys"
        assert "ego" in self.read_keys, "ego should be in read_keys"

        bev = self._load_bev(index, "bev_world")["bev"]
        bev_vehicle = bev[-2]
        bev_vehicle_sum = bev_vehicle.sum()
        bev_vehicle_sum = (bev_vehicle_sum - 1500) / 1500  # 2600

        bev_road_red_yellow = bev[3]
        bev_road_red_yellow_sum = bev_road_red_yellow.sum()
        bev_road_red_yellow_sum = (bev_road_red_yellow_sum - 4000) / 4000  # 3000

        bev_road_green = bev[4]
        bev_road_green_sum = bev_road_green.sum()
        bev_road_green_sum = (bev_road_green_sum - 3000) / 3000  # 2300

        yaw = self._load_json(index, "ego")["rotation_array"][2]
        # Normalize to 0-360
        yaw = yaw % 90 - 45
        yaw = np.abs(yaw) / 22.5

        weight_vehicle = scipy.stats.norm(0, 1).pdf(bev_vehicle_sum) * 5
        weight_road_red_yellow = scipy.stats.norm(0, 1).pdf(bev_road_red_yellow_sum)
        weight_road_green = scipy.stats.norm(0, 1).pdf(bev_road_green_sum)
        weight_yaw = scipy.stats.norm(0, 1).pdf(yaw) * 10  # 2

        return weight_vehicle * weight_road_red_yellow * weight_road_green * weight_yaw

    def _load_bev(self, index, read_key):
        load_path = self.data[index][0] / read_key / f"{self.data[index][1]}.npz"
        data = np.load(load_path)
        bev_ = data["bev"]
        bev_ = torch.from_numpy(bev_).float()
        # Permute the dimensions such that the channel dim is the first one
        agent_mask = bev_[..., self.bev_agent_channel]
        bev_[..., self.bev_vehicle_channel] = torch.logical_and(
            bev_[..., self.bev_vehicle_channel],
            torch.logical_not(bev_[..., self.bev_agent_channel]),
        )

        bev = bev_[..., self.bev_selected_channels]

        bev = bev.permute(2, 0, 1)
        # Add offroad mask to BEV representation
        if self.bev_calculate_offroad:
            offroad_mask = torch.where(
                torch.all(bev == 0, dim=0),
                torch.ones_like(bev[0]),
                torch.zeros_like(bev[0]),
            )
            bev = torch.cat([bev, offroad_mask.unsqueeze(0)], dim=0)

        return {"bev": bev, "agent_mask": agent_mask}

    def _load_rgb(self, index, read_key):
        load_path = self.data[index][0] / read_key / f"{self.data[index][1]}.png"
        image = cv2.imread(str(load_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        return image

    def _load_json(self, index, read_key):

        load_path = self.data[index][0] / read_key / f"{self.data[index][1]}.json"
        ego = json.load(open(load_path))
        ego_ = {}
        for (key, value) in ego.items():
            if value != "<<??>>" and not isinstance(value, str):
                ego_[key] = torch.tensor(ego[key], dtype=torch.float32)
        return ego_

    def set_default_config(self):
        self.config = {
            "data_path": None,
            "sequence_length": 20,
            "read_keys": ["bev_world", "ego", "navigation"],
            "dilation": 1,
            "bev_calculate_offroad": True,
            "bev_selected_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "bev_vehicle_channel": 1,
            "bev_agent_channel": 2,
        }

    def append_config(self, config):
        self.config.update(config)


class InstanceDatasetRAM(Dataset):
    """This dataset is very similar to InstanceWriter. It reads the data
    from different folders and appends to a Dict."""

    def __init__(
        self,
        config: dict,
    ):

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.count_array = []
        self.data = []

        for item in os.listdir(self.data_path):

            if item.startswith("episode"):

                episode_path = self.data_path / item

                print(f"Reading {episode_path} ...")

                key_lengths = np.array(
                    [
                        len(os.listdir(episode_path / read_key))
                        for read_key in self.read_keys
                    ]
                )
                if not np.all(key_lengths == key_lengths[0]):
                    print("All keys should include same amount of data!")
                    print(f"Skipping {episode_path}")

                key_ = self.read_keys[0]

                episode_key_path = episode_path / key_

                step_list = sorted(
                    [int(x.split(".")[0]) for x in os.listdir(episode_key_path)]
                )

                counter = 0
                for step in step_list:

                    data = {}

                    for read_key in self.read_keys:

                        if read_key in ["bev", "bev_world", "bev_ego"]:

                            data_ = self._load_bev(episode_path, step, read_key)

                        if read_key in ["rgb_front", "rgb_left", "rgb_right"]:

                            data_ = self._load_rgb(episode_path, step, read_key)

                        if read_key in [
                            "ego",
                            "navigation",
                            "occ",
                            "navigation_downsampled",
                        ]:

                            data_ = self._load_json(episode_path, step, read_key)

                        data[read_key] = data_

                    self.data.append([episode_path, step, data])
                    counter += 1

                self.count_array.append(
                    counter + (self.count_array[-1] if len(self.count_array) > 0 else 0)
                )

        self.count_array.pop(-1)
        self.count_array = np.array(self.count_array)

    def build_from_config(self):

        self.data_path = Path(self.config["data_path"])
        self.sequence_length = self.config["sequence_length"]
        self.read_keys = self.config["read_keys"]
        self.dilation = self.config["dilation"]
        self.bev_agent_channel = self.config["bev_agent_channel"]
        self.bev_vehicle_channel = self.config["bev_vehicle_channel"]
        self.bev_selected_channels = self.config["bev_selected_channels"]
        self.bev_calculate_offroad = self.config["bev_calculate_offroad"]

    def __len__(self):
        return (
            len(self.data)
            - self.sequence_length
            - (self.sequence_length - 1) * (self.dilation - 1)
            + 1
        )

    def __getitem__(self, index):

        (I,) = np.nonzero(
            np.logical_and(
                (
                    (
                        (index + self.sequence_length * self.dilation - 1)
                        - self.count_array
                    )
                    >= 0
                ),
                (
                    (
                        (index + self.sequence_length * self.dilation - 1)
                        - self.count_array
                    )
                    <= self.sequence_length * self.dilation - 1
                ),
            )
        )

        if I.size != 0:
            index = self.count_array[I[-1]]

        data_sample = self.data[index][2]

        data = {}

        for read_key in self.read_keys:

            if isinstance(data_sample[read_key], dict):

                data_ = {}

                for key in data_sample[read_key].keys():

                    data_[key] = torch.stack(
                        [
                            self.data[index][2][read_key][key]
                            for index in range(
                                index,
                                index + self.sequence_length * self.dilation,
                                self.dilation,
                            )
                        ],
                        dim=0,
                    )

            else:

                data_ = torch.stack(
                    [
                        self.data[index][2][read_key]
                        for index in range(
                            index,
                            index + self.sequence_length * self.dilation,
                            self.dilation,
                        )
                    ],
                    dim=0,
                )

            # data = default_collate(
            #     [
            #         self.data[index][2]
            #         for index in range(
            #             index,
            #             index + self.sequence_length * self.dilation,
            #             self.dilation,
            #         )
            #     ]
            # )

            data[read_key] = data_

        return data

    def __getweight__(self, index):

        assert "bev_world" in self.read_keys, "bev_world should be in read_keys"
        assert "ego" in self.read_keys, "ego should be in read_keys"

        bev = self._load_bev(index, "bev_world")["bev"]
        bev_vehicle = bev[-2]
        bev_vehicle_sum = bev_vehicle.sum()
        bev_vehicle_sum = (bev_vehicle_sum - 1500) / 1500  # 2600

        bev_road_red_yellow = bev[3]
        bev_road_red_yellow_sum = bev_road_red_yellow.sum()
        bev_road_red_yellow_sum = (bev_road_red_yellow_sum - 4000) / 4000  # 3000

        bev_road_green = bev[4]
        bev_road_green_sum = bev_road_green.sum()
        bev_road_green_sum = (bev_road_green_sum - 3000) / 3000  # 2300

        yaw = self._load_json(index, "ego")["rotation_array"][2]
        # Normalize to 0-360
        yaw = yaw % 90 - 45
        yaw = np.abs(yaw) / 22.5

        weight_vehicle = scipy.stats.norm(0, 1).pdf(bev_vehicle_sum) * 5
        weight_road_red_yellow = scipy.stats.norm(0, 1).pdf(bev_road_red_yellow_sum)
        weight_road_green = scipy.stats.norm(0, 1).pdf(bev_road_green_sum)
        weight_yaw = scipy.stats.norm(0, 1).pdf(yaw) * 10  # 2

        return weight_vehicle * weight_road_red_yellow * weight_road_green * weight_yaw

    def _load_bev(self, episode_path, step, read_key):
        load_path = episode_path / read_key / f"{step}.npz"
        data = np.load(load_path)
        bev_ = data["bev"]
        bev_ = torch.from_numpy(bev_).float()
        # Permute the dimensions such that the channel dim is the first one
        agent_mask = bev_[..., self.bev_agent_channel]
        bev_[..., self.bev_vehicle_channel] = torch.logical_and(
            bev_[..., self.bev_vehicle_channel],
            torch.logical_not(bev_[..., self.bev_agent_channel]),
        )

        bev = bev_[..., self.bev_selected_channels]

        bev = bev.permute(2, 0, 1)
        # Add offroad mask to BEV representation
        if self.bev_calculate_offroad:
            offroad_mask = torch.where(
                torch.all(bev == 0, dim=0),
                torch.ones_like(bev[0]),
                torch.zeros_like(bev[0]),
            )
            bev = torch.cat([bev, offroad_mask.unsqueeze(0)], dim=0)

        return {"bev": bev, "agent_mask": agent_mask}

    def _load_rgb(self, episode_path, step, read_key):
        load_path = episode_path / read_key / f"{step}.png"
        image = cv2.imread(str(load_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        return image

    def _load_json(self, episode_path, step, read_key):

        load_path = episode_path / read_key / f"{step}.json"
        ego = json.load(open(load_path))
        ego_ = {}
        for (key, value) in ego.items():
            if value != "<<??>>" and not isinstance(value, str):
                ego_[key] = torch.tensor(ego[key], dtype=torch.float32)
        return ego_

    def set_default_config(self):
        self.config = {
            "data_path": None,
            "sequence_length": 30,
            "read_keys": ["bev_world", "ego", "navigation", "occ"],
            "dilation": 1,
            "bev_calculate_offroad": False,
            "bev_selected_channels": [0, 1, 2, 3, 4, 5, 6, 11],
            "bev_vehicle_channel": 6,
            "bev_agent_channel": 7,
        }

    def append_config(self, config):
        self.config.update(config)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Testing InstanceDatasetRAM")

    dataset = InstanceDatasetRAM(
        config={
            "data_path": "/home/volkan/Documents/Codes/carla_env/data/ground_truth_bev_model_dummy_data_20Hz_multichannel_bev_dense_traffic/"
        }
    )

    pass
# if __name__ == "__main__":

#     dataset = InstanceDataset(
#         data_path="/home/volkan/Documents/Codes/carla_env/data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33",
#         sequence_length=20,
#         read_keys=["occ"],
#         dilation=3,
#     )
#     dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)

#     for k in range(len(dataset)):
#         print(k)

#     print(len(dataset))
