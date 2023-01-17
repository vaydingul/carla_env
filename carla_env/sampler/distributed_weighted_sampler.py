import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math


class DistributedWeightedSampler(Sampler):
    def __init__(
            self,
            dataset,
            weights=None,
            num_replicas=None,
            rank=None,
            replacement=True,
            shuffle=True):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # print(f"Rank {self.rank} has {self.num_samples} samples. Total size is {self.total_size}")
        self.replacement = replacement
        self.shuffle = shuffle
        self.weights = weights

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.weights is not None:
            # print(f"Rank {self.rank} has {len(self.weights)} weights. Length of weight is {len(self.weights)}")
            # print(f"Rank {self.rank} has {len(self.dataset)} dataset. Length of dataset is {len(self.dataset)}")
            assert len(self.weights) == len(self.dataset), f"Weights ({len(self.weights)}) must be the same length as the dataset ({len(self.dataset)})"

            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=g).tolist()
            
        else:

            if self.shuffle:
                indices = torch.randperm(
                    len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
        # print(f"Rank {self.rank} has {len(indices)} indices. Max index is {max(indices)}")
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # print(f"Rank {self.rank} has {len(indices)} indices. Max index is {max(indices)}")
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # print(f"Rank {self.rank} has {len(indices)} indices. Max index is {max(indices)}")
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
