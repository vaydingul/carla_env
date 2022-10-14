import h5py
import json


class InstanceWriter(Object):
    # TODO: Make add_key function to organize which keys are saved as what
    # TODO: Define some common "saving" types. For example, "image -> folder", "structured_data -> json"
    # TODO: Writer take "step_count" and "general data dict" as inputs to write function. For each key, it will
    # maintain the required operations
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def write(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)


class H5DFWriter(Object):
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def write(self):
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('data', data=self.data)
