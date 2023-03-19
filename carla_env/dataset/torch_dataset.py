import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, default_collate
import torch


class TorchDataset(Dataset):
    """This dataset is very similar to InstanceWriter. It reads the data
    from different folders and appends to a Dict."""

    def __init__(
        self,
        config: dict,
    ):

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.data = []
        self.length = 0
        for item in os.listdir(self.data_path):

            if item.endswith(".pth"):

                self.length += 1

    def set_default_config(self):

        self.config = {
            "data_path": None,
        }

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        data = torch.load(self.data_path / f"{idx}.pth")

        return data

    def append_config(self, config):
        self.config.update(config)

    def build_from_config(self):

        self.data_path = Path(self.config["data_path"])


if __name__ == "__main__":

    dataset = TorchDataset(
        config={
            "data_path": "/home/volkan/Documents/Codes/carla_env/data/ground_truth_bev_model_dummy_data_20Hz_multichannel_bev_dense_traffic_converted"
        }
    )

    for data in dataset:

        pass
