import h5py
import json



class InstanceWriter(Object):
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
		with h5df.File(self.path, 'w') as f:
			f.create_dataset('data', data=self.data)