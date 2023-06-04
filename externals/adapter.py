from abc import ABC, abstractmethod
import torch


class Adapter(ABC):
    def __init__(self, config):
        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

    @abstractmethod
    def build_from_config(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def set_default_config(self):
        pass

    def append_config(self, config):
        self.config.update(config)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                self.__dict__[k] = v.to(device)
