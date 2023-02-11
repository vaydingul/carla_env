import json
import os
from enum import Enum, auto
from pathlib import Path

import cv2
import h5py
import numpy as np


def DEFAULT_JSON(o):
    return f"<<??>>"


class InstanceWriterType(Enum):
    """Type of instance writer."""
    JSON = auto()
    RGB_IMAGE = auto()
    BEV_IMAGE = auto()


class InstanceWriter:

    def __init__(self, path):

        self.path = Path(path)
        self.data_dict = {}

    def add_key(
            self,
            key: str,
            value: str,
            type: InstanceWriterType = InstanceWriterType.JSON):
        """ Add a key to the data dict """
        os.makedirs(self.path / key, exist_ok=True)

        self.data_dict[key] = (value, type)

    def write(self, count, data):

        # Check if all keys exist
        ALL_KEYS_EXIST = True
        for key in self.data_dict.keys():
            ALL_KEYS_EXIST = ALL_KEYS_EXIST and (key in data.keys())

        if ALL_KEYS_EXIST:

            for key, (value, type) in self.data_dict.items():

                if type == InstanceWriterType.JSON:

                    with open(self.path / key / f"{count}.json", 'w') as f:

                        json.dump(
                            obj=data[value],
                            fp=f,
                            indent=10,
                            default=DEFAULT_JSON)

                elif type == InstanceWriterType.RGB_IMAGE:

                    # Take the image
                    image = data[value]["data"]

                    # Convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Save the image
                    cv2.imwrite(str(self.path / key / f"{count}.png"), image)

                elif type == InstanceWriterType.BEV_IMAGE:

                    # Take the image
                    bev = data[value]

                    # Save the BEV
                    np.savez(str(self.path / key / f"{count}.npz"), bev=bev)
