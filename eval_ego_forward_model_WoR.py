import numpy as np
import matplotlib.pyplot as plt
from carla_env.models.dynamic.vehicle_WoR import EgoModel
#from ego_model import EgoModel
import os 
import pathlib
import torch
import argparse
from utils.plot_utils import plot_result_eval
import time

def evaluate(fname, data, model, config):
	

	vehicle_location = torch.Tensor(data["vehicle_location"])
	vehicle_rotation = torch.Tensor(data["vehicle_rotation"])
	vehicle_velocity = torch.Tensor(data["vehicle_velocity"])
	vehicle_control = torch.Tensor(data["vehicle_control"])
	elapsed_time = torch.Tensor(data["elapsed_time"])
	
	print(f"Total Elapsed Time: {elapsed_time[-1]}")
	print(f"Number of Steps: {elapsed_time.shape[0]}")

	location_predicted = []
	yaw_predicted = []
	speed_predicted = []

	for k in range(0, elapsed_time.shape[0]-1):
		
		if k % 10 == 0:

			location = vehicle_location[k, :2]
			yaw = vehicle_rotation[k, 1:2]
			speed = torch.norm(vehicle_velocity[k, :], dim = -1, keepdim = True)
		
			location_predicted.append(location)
			yaw_predicted.append(yaw)
			speed_predicted.append(speed)

			continue

		action = vehicle_control[k, :]

		location, yaw, speed = model(location, yaw, speed, action)

		location_predicted.append(location)
		yaw_predicted.append(yaw)
		speed_predicted.append(speed)

	location_predicted = torch.stack(location_predicted, dim = 0).detach().numpy()
	yaw_predicted = torch.stack(yaw_predicted, dim = 0).detach().numpy()
	speed_predicted = torch.stack(speed_predicted, dim = 0).detach().numpy()

	vehicle_location = vehicle_location.numpy()
	vehicle_rotation = vehicle_rotation.numpy()
	
	location_loss = np.mean(np.abs(vehicle_location[:-1, :2] - location_predicted))
	rotation_loss = np.mean(np.abs(np.cos(vehicle_rotation[:-1, 1:2]) - np.cos(yaw_predicted)))
	rotation_loss += np.mean(np.abs(np.sin(vehicle_rotation[:-1, 1:2]) - np.sin(yaw_predicted)))


	savedir = pathlib.Path(f"figures/ego-forward-model-evaluation-WoR-2/{fname}/")
	os.makedirs(savedir, exist_ok=True)

	if config.plot_local:
		plot_result_eval(vehicle_location, vehicle_rotation, vehicle_velocity, vehicle_control, elapsed_time, location_predicted, yaw_predicted, speed_predicted, savedir)
		time.sleep(1)

	return location_loss, rotation_loss
	
def main(config):
	folder_name = config.evaluation_data_folder
	

	model = EgoModel(dt = 1/20)
	model.load_state_dict(torch.load(config.model_path))
	model.eval()


	location_loss_list = []
	rotation_loss_list = []

	for file in os.listdir(folder_name):
		if file.endswith(".npz"):
			file_path = os.path.join(folder_name, file)
			data = np.load(file_path)
			
			location_loss, rotation_loss = evaluate(file, data, model, config)
			location_loss_list.append(location_loss)
			rotation_loss_list.append(rotation_loss)
	
	print(f"Average Location Loss: {np.mean(location_loss_list)}")
	print(f"Average Rotation Loss: {np.mean(rotation_loss_list)}")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, default="pretrained_models/WoR/ego_model.th")
	parser.add_argument("--evaluation_data_folder", type=str, default="data/kinematic_model_data_val_2")
	parser.add_argument("--wandb", type=bool, default=False)
	parser.add_argument("--plot_local", type=bool, default=True)
	config = parser.parse_args()

	main(config)




