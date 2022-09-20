import torch
from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
from carla_env.models.dynamic import vehicle as v

class EgoModelDataset(Dataset):

	def __init__(self, data_path, rollout_length = 10):
		

		self.location = []
		self.rotation = []
		self.velocity = []
		self.acceleration = []
		self.control = []
		self.elapsed_time = []


		self.rollout_length = rollout_length

		for file in os.listdir(data_path):
			if file.endswith(".npz"):
				data_ = np.load(os.path.join(data_path, file))
				self.location.append(data_["vehicle_location"])
				self.rotation.append(data_["vehicle_rotation"])
				self.velocity.append(data_["vehicle_velocity"])
				self.acceleration.append(data_["vehicle_acceleration"])
				self.control.append(data_["vehicle_control"])
				self.elapsed_time.append(data_["elapsed_time"])

		self.location = torch.Tensor(np.concatenate(self.location, axis = 0))
		self.rotation = torch.Tensor(np.concatenate(self.rotation, axis = 0))
		self.velocity = torch.Tensor(np.concatenate(self.velocity, axis = 0))
		self.acceleration = torch.Tensor(np.concatenate(self.acceleration, axis = 0))
		self.control = torch.Tensor(np.concatenate(self.control, axis = 0))
		self.elapsed_time = torch.Tensor(np.concatenate(self.elapsed_time, axis = 0))
	
	def __len__(self) -> int:
		return self.elapsed_time.shape[0] - self.rollout_length

	def __getitem__(self, index):

		return self.location[index: index + self.rollout_length, :], self.rotation[index: index + self.rollout_length, :], self.velocity[index: index + self.rollout_length, :], self.acceleration[index: index + self.rollout_length, :], self.control[index: index + self.rollout_length, :], self.elapsed_time[index: index + self.rollout_length]

class Learner(object):

	def __init__(self, model, optimizer, loss_criterion, device, data_loader, num_epochs = 1000, log_interval = 10):
		self.model = model
		self.optimizer = optimizer
		self.loss_criterion = loss_criterion
		self.device = device
		self.data_loader = data_loader
		self.num_epochs = num_epochs
		self.log_interval = log_interval
	
	def train(self):
		pass

	def validate(self):
		pass

	def learn(self):
		pass



if __name__ == "__main__":
	
	data_path = "./data/kinematic_model_data/"
	ego_model_dataset = EgoModelDataset(data_path)
	ego_model_dataloader = DataLoader(ego_model_dataset, batch_size = 1, shuffle = True, num_workers = 0)


	ego_model = v.KinematicBicycleModel()
	ego_model_optimizer = torch.optim.Adam(ego_model.parameters(), lr = 0.001)
	ego_model_loss_criterion = torch.nn.MSELoss()
	ego_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	ego_model.to(ego_model_device)

	ego_model_learner = Learner(ego_model, ego_model_optimizer, ego_model_loss_criterion, ego_model_device, ego_model_dataloader)

	ego_model_learner.learn()

	# for i, (location, rotation, velocity, acceleration, control, elapsed_time) in enumerate(ego_model_dataloader):
	# 	print(location.shape)
	# 	print(rotation.shape)
	# 	print(velocity.shape)
	# 	print(acceleration.shape)
	# 	print(control.shape)
	# 	print(elapsed_time.shape)
	# 	break