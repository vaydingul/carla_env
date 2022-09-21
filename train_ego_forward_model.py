import torch
from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
from carla_env.models.dynamic import vehicle as v
import wandb
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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

	def __init__(self, model, dataloader_train, dataloader_val, optimizer, loss_criterion, device, num_epochs = 1000, log_interval = 10):
		self.model = model
		self.dataloader_train = dataloader_train
		self.dataloader_val = dataloader_val
		self.optimizer = optimizer
		self.loss_criterion = loss_criterion
		self.device = device
		self.num_epochs = num_epochs
		self.log_interval = log_interval
	
	def train(self, epoch, run):
		
		self.model.train()

		counter = 0

		for i, (location, rotation, velocity, _, control, _) in enumerate(self.dataloader_train):

			location = location[..., :2].to(self.device)
			rotation = rotation[..., 1:2].to(self.device)
			#velocity = velocity[..., :1].to(self.device)
			velocity = torch.norm(velocity, dim = -1, keepdim = True).to(self.device)
			control = control.to(self.device)

			location_pred = []
			rotation_pred = []
			velocity_pred = []


			location_ = location[:, 0]
			rotation_ = rotation[:, 0]
			velocity_ = velocity[:, 0]

			
			for t in range(location.shape[1] - 1):

				control_ = control[:, t]

				location_, rotation_, velocity_ = self.model(location_, rotation_, velocity_, control_)
				
				location_pred.append(location_)
				rotation_pred.append(rotation_)
				velocity_pred.append(velocity_)

			location_pred = torch.stack(location_pred, dim = 1)
			rotation_pred = torch.stack(rotation_pred, dim = 1)
			velocity_pred = torch.stack(velocity_pred, dim = 1)

			loss_location = self.loss_criterion(location[:, 1:], location_pred)
			loss_rotation = self.loss_criterion(torch.cos(rotation[:, 1:]), torch.cos(rotation_pred))
			loss_rotation += self.loss_criterion(torch.sin(rotation[:, 1:]), torch.sin(rotation_pred))

			loss = loss_location + loss_rotation
			
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			step = epoch * len(self.dataloader_train.dataset) + counter * self.dataloader_train.batch_size + location.shape[0]
			run.log({"train/step" : step, "train/loss" : loss, "train/loss_location" : loss_location, "train/loss_rotation" : loss_rotation})
			#print("Iter: {}, Loss: {}".format(i, loss.item()))

	def validate(self, epoch, run = None):
		
		self.model.eval()

		losses_total = []
		losses_location = []
		losses_rotation = []

		counter = 0
		with torch.no_grad():

			for i, (location, rotation, velocity, _, control, _) in enumerate(self.dataloader_val):

				location = location[..., :2].to(self.device)
				rotation = rotation[..., 1:2].to(self.device)
				#velocity = velocity[..., 0:1].to(self.device)
				velocity = torch.norm(velocity, dim = -1, keepdim = True).to(self.device)
				control = control.to(self.device)

				location_pred = []
				rotation_pred = []
				velocity_pred = []


				location_ = location[:, 0]
				rotation_ = rotation[:, 0]
				velocity_ = velocity[:, 0]

				
				for t in range(location.shape[1] - 1):

					control_ = control[:, t]

					location_, rotation_, velocity_ = self.model(location_, rotation_, velocity_, control_)
					
					location_pred.append(location_)
					rotation_pred.append(rotation_)
					velocity_pred.append(velocity_)

				location_pred = torch.stack(location_pred, dim = 1)
				rotation_pred = torch.stack(rotation_pred, dim = 1)
				velocity_pred = torch.stack(velocity_pred, dim = 1)


				loss_location = self.loss_criterion(location_pred, location[:, 1:])
				loss_rotation = self.loss_criterion(torch.cos(rotation_pred), torch.cos(rotation[:, 1:]))
				loss_rotation += self.loss_criterion(torch.sin(rotation_pred), torch.sin(rotation[:, 1:]))

				loss = loss_location + loss_rotation
				
				losses_total.append(loss.item())
				losses_location.append(loss_location.item())
				losses_rotation.append(loss_rotation.item())

				step = epoch * len(self.dataloader_val.dataset) + counter * self.dataloader_val.batch_size + location.shape[0]
				counter += 1

		loss = np.mean(losses_total)
		loss_location = np.mean(losses_location)
		loss_rotation = np.mean(losses_rotation)

		run.log({"val/step" : step, "val/loss" : loss, "val/loss_location" : loss_location, "val/loss_rotation" : loss_rotation})
		run.log({"model/step": step, "model/front_wheelbase": self.model.front_wheelbase.item(), "model/rear_wheelbase": self.model.rear_wheelbase.item(), "model/steer_gain": self.model.steer_gain.item(), "model/brake_acceleration": self.model.brake_acceleration.item()})
		return loss, loss_location, loss_rotation

	def learn(self, run = None):
		
		for epoch in range(self.num_epochs):
			
			self.train(epoch, run)
			loss, loss_location, loss_orientation = self.validate(epoch, run)
			logger.info("Epoch: {}, Val Loss: {}, Val Loss Location: {}, Val Loss Orientation: {}".format(epoch, loss, loss_location, loss_orientation))
			



def main(config):
	

	data_path_train = config.data_path_train
	data_path_val = config.data_path_val
	ego_model_dataset_train = EgoModelDataset(data_path_train)
	ego_model_dataset_val = EgoModelDataset(data_path_val)
	logger.info(f"Train dataset size: {len(ego_model_dataset_train)}")
	logger.info(f"Validation dataset size: {len(ego_model_dataset_val)}")

	ego_model_dataloader_train = DataLoader(ego_model_dataset_train, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers)
	ego_model_dataloader_val = DataLoader(ego_model_dataset_val, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers)

	ego_model = v.KinematicBicycleModel(dt = 1/20)
	ego_model_optimizer = torch.optim.Adam(ego_model.parameters(), lr = config.lr)
	ego_model_loss_criterion = torch.nn.L1Loss()
	ego_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(ego_model_device)
	ego_model.to(ego_model_device)

	ego_model_learner = Learner(ego_model, ego_model_dataloader_train, ego_model_dataloader_val, ego_model_optimizer, ego_model_loss_criterion, ego_model_device, num_epochs=config.num_epochs)

	run = wandb.init(project = "mbl", name = "ego-forward-model-training", config = config)
	run.define_metric("train/step")
	run.define_metric("val/step")
	run.define_metric("model/step")
	run.define_metric(name = "train/*", step_metric = "train/step")
	run.define_metric(name = "val/*", step_metric = "val/step")
	run.define_metric(name = "model/*", step_metric = "model/step")
	ego_model_learner.learn(run)
	
	ego_model.to("cpu")
	torch.save(ego_model.state_dict(), "ego_model_new.pt")


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type = float, default = 0.01)
	parser.add_argument("--num_epochs", type = int, default = 100)
	parser.add_argument("--batch_size", type = int, default = 1000)
	parser.add_argument("--num_workers", type = int, default = 0)
	parser.add_argument("--data_path_train", type = str, default = "./data/kinematic_model_data_train_2/")
	parser.add_argument("--data_path_val", type = str, default = "./data/kinematic_model_data_val_2/")
	
	config = parser.parse_args()
	
	

	main(config)
	
