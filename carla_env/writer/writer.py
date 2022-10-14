import json
import os
from pathlib import Path
from enum import Enum, auto

import h5py
import cv2


class InstanceWriterType(Enum):
    """Type of instance writer."""
    JSON = auto()
    IMAGE = auto()


class InstanceWriter:
    # TODO: Make add_key function to organize which keys are saved as what
    # TODO: Define some common "saving" types. For example, "image -> folder", "structured_data -> json"
    # TODO: Writer take "step_count" and "general data dict" as inputs to write function. For each key, it will
    # maintain the required operations
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

        for key, (value, type) in self.data_dict.items():

            if type == InstanceWriterType.JSON:

                with open(self.path / key / f"{count}.json", 'w') as f:

                    json.dump(obj=data[value], fp=f, indent=10)

            if type == InstanceWriterType.IMAGE:

                # Take the image
                image = data[value]

                # Convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Save the image
                cv2.imwrite(str(self.path / key / f"{count}.png"), image)


# class H5DFWriter(Object):
#     def __init__(self, data, path):
#         self.data = data
#         self.path = path

#     def write(self):
#         with h5py.File(self.path, 'w') as f:
#             f.create_dataset('data', data=self.data)
