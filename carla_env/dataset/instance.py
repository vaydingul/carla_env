
import torch
from pathlib import Path
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Union


class InstanceDataset(Dataset):
    """This dataset is very similar to InstanceWriter. It reads the data
     from different folders and appends to a Dict."""

    def __init__(
            self,
            data_path: Union[str, Path],
            sequence_length: int = 2,
            read_keys: List[str] = ["bev"]):

        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.read_keys = read_keys
        self.count_array = []
        self.data = []

        for item in os.listdir(data_path):

            if item.startswith("episode"):

                episode_path = self.data_path / item

                key_lengths = np.array([len(os.listdir(episode_path / read_key))
                                        for read_key in read_keys])
                assert np.all(
                    key_lengths == key_lengths[0]), "All keys should include same amount of data!"

                key_ = read_keys[0]

                episode_key_path = episode_path / key_

                step_list = os.listdir(episode_key_path)
                step_list.sort(key=lambda x: int(x.split('.')[0]))

                counter = 0
                for step in step_list:

                    self.data.append([episode_path, step])
                    counter += 1

                self.count_array.append(
                    counter +
                    (self.count_array[-1] if len(self.count_array) > 0 else 0))

        self.count_array = np.array(self.count_array)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):

        I, = np.nonzero(np.logical_and((((index +
                                          self.sequence_length) -
                                         self.count_array) >= 0), (((index +
                                                                     self.sequence_length) -
                                                                    self.count_array) < self.sequence_length)))

        if I.size != 0:
            index = self.count_array[I[-1]]

        data = {}

        for read_key in self.read_keys:

            if read_key == "bev":

                data_ = torch.stack([self._load_bev(index + k)
                                     for k in range(self.sequence_length)],
                                    dim=0)

            if (read_key == "rgb_front" or read_key ==
                            "rgb_right" or read_key == "rgb_left"):

                data_ = self._load_rgb(index, read_key)

            if read_key == "ego":

                data_ = self._load_ego(index)

            data[read_key] = data_

        return data

    def _load_bev(self, index):

        load_path = self.data[index][0] / "bev" / self.data[index][1]
        data = np.load(load_path)
        bev = data["bev"]
        bev = torch.from_numpy(bev).float()
        # Permute the dimensions such that the channel dim is the first one
        bev = bev[..., [k for k in range(bev.shape[-1]) if k != 3]]
        bev = bev.permute(2, 0, 1)

        return bev

    def _load_rgb(self, index, read_key):
        pass

    def _load_ego(self, index):
        pass


if __name__ == "__main__":

    dataset = InstanceDataset(
        data_path="data/ground_truth_bev_model_train_data")
    pass
