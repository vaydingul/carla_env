import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from typing import List, Union
import cv2
import json
import logging

logger = logging.getLogger(__name__)


class InstanceDataset(Dataset):
    """This dataset is very similar to InstanceWriter. It reads the data
     from different folders and appends to a Dict."""

    def __init__(
            self,
            data_path: Union[str, Path],
            sequence_length: int = 2,
            read_keys: List[str] = ["bev_world"],
            dilation: int = 1):

        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.read_keys = read_keys
        self.dilation = dilation
        self.count_array = []
        self.data = []

        for item in os.listdir(data_path):

            if item.startswith("episode"):

                episode_path = self.data_path / item

                key_lengths = np.array([len(os.listdir(episode_path / read_key))
                                        for read_key in read_keys])
                if not np.all(
                        key_lengths == key_lengths[0]):
                    logger.info("All keys should include same amount of data!")
                    logger.info(f"Skipping {episode_path}")

                key_ = read_keys[0]

                episode_key_path = episode_path / key_

                step_list = sorted([int(x.split('.')[0])
                                   for x in os.listdir(episode_key_path)])

                counter = 0
                for step in step_list:

                    self.data.append([episode_path, step])
                    counter += 1

                self.count_array.append(
                    counter +
                    (self.count_array[-1] if len(self.count_array) > 0 else 0))

        self.count_array.pop(-1)
        self.count_array = np.array(self.count_array)

    def __len__(self):
        return len(self.data) - self.sequence_length - \
            (self.sequence_length - 1) * (self.dilation - 1) + 1

    def __getitem__(self, index):

        I, = np.nonzero(np.logical_and((((index +
                                          self.sequence_length *
                                          self.dilation) -
                                         self.count_array) >= 0), (((index +
                                                                     self.sequence_length *
                                                                     self.dilation) -
                                                                    self.count_array) <= self.sequence_length *
                                                                   self.dilation)))

        if I.size != 0:
            index = self.count_array[I[-1]]

        data = {}

        for read_key in self.read_keys:

            if read_key in ["bev", "bev_world", "bev_ego"]:

                data_ = [
                    self._load_bev(
                        index +
                        k,
                        read_key) for k in range(
                        0,
                        self.sequence_length *
                        self.dilation,
                        self.dilation)]

                data_stacked = {}

                for key in data_[0].keys():
                    data_stacked[key] = torch.stack(
                        [data_[k][key] for k in range(len(data_))], dim=0)

                data_ = data_stacked

            if read_key in ["rgb_front", "rgb_left", "rgb_right"]:

                data_ = torch.stack(
                    [
                        self._load_rgb(
                            index +
                            k,
                            read_key) for k in range(
                            0,
                            self.sequence_length *
                            self.dilation,
                            self.dilation)],
                    dim=0)

            if read_key in [
                "ego",
                "navigation",
                "occ",
                    "navigation_downsampled"]:

                data_ = [
                    self._load_json(
                        index +
                        k,
                        read_key) for k in range(
                        0,
                        self.sequence_length *
                        self.dilation,
                        self.dilation)]
                data_stacked = {}

                for key in data_[0].keys():
                    data_stacked[key] = torch.stack(
                        [data_[k][key] for k in range(len(data_))], dim=0)
                data_ = data_stacked

            data[read_key] = data_

        return data

    def _load_bev(self, index, read_key):

        load_path = self.data[index][0] / \
            read_key / f"{self.data[index][1]}.npz"
        data = np.load(load_path)
        bev_ = data["bev"]
        bev_ = torch.from_numpy(bev_).float()
        # Permute the dimensions such that the channel dim is the first one
        agent_mask = bev_[..., 3]
        bev_[..., 2] -= agent_mask
        bev = bev_[..., [k for k in range(bev_.shape[-1]) if k != 3]]

        bev = bev.permute(2, 0, 1)
        # Add offroad mask to BEV representation
        offroad_mask = torch.where(
            torch.all(
                bev == 0, dim=0), torch.ones_like(
                bev[0]), torch.zeros_like(
                bev[0]))
        bev = torch.cat([bev, offroad_mask.unsqueeze(0)], dim=0)

        return {"bev": bev,
                "agent_mask": agent_mask}

    def _load_rgb(self, index, read_key):
        load_path = self.data[index][0] / \
            read_key / f"{self.data[index][1]}.png"
        image = cv2.imread(str(load_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        return image

    def _load_json(self, index, read_key):

        load_path = self.data[index][0] / \
            read_key / f"{self.data[index][1]}.json"
        ego = json.load(open(load_path))
        ego_ = {}
        for (key, value) in ego.items():
            if value != "<<??>>" and not isinstance(value, str):
                ego_[key] = torch.tensor(ego[key], dtype=torch.float32)
        return ego_


if __name__ == "__main__":

    dataset = InstanceDataset(
        data_path="/home/vaydingul/Documents/Codes/carla_env/data/ground_truth_bev_model_dummy_data",
        sequence_length=20,
        read_keys=["occ"],
        dilation=3)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0)
    k = 1
    for data in dataloader:
        print(k)
        k += 1
    print(len(dataloader))
