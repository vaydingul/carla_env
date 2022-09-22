import numpy as np
import matplotlib.pyplot as plt
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
import os 
import pathlib
import torch
import argparse
from utils.plot_utils import plot_result


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

	for k in range(1, elapsed_time.shape[0]):
		
		location = vehicle_location[k-1, :2]
		yaw = vehicle_rotation[k-1, 1]
		speed = torch.norm(vehicle_velocity[k-1, :], dim = -1, keepdim = True)
		action = vehicle_control[k-1, :]

		location_, yaw_, speed_ = model(location, yaw, speed, action)

		location_predicted.append(location_)
		yaw_predicted.append(yaw_)
		speed_predicted.append(speed_)

	location_predicted = torch.stack(location_predicted, dim = 0).detach().numpy()
	yaw_predicted = torch.stack(yaw_predicted, dim = 0).detach().numpy()
	speed_predicted = torch.stack(speed_predicted, dim = 0).detach().numpy()

	vehicle_location = vehicle_location.numpy()
	
		

	savedir = pathlib.Path(f"figures/ego-forward-model-evaluation/{fname}/")
	os.makedirs(savedir, exist_ok=True)

	if config.plot_local:
		plot_result(vehicle_location, vehicle_rotation, vehicle_velocity, vehicle_control, elapsed_time, location_predicted, yaw_predicted, speed_predicted, savedir)
	

def main(config):
	folder_name = config.evaluation_data_folder
	

	model = KinematicBicycleModel()
	model.state_dict = torch.load(config.model_path)
	model.eval()

	for file in os.listdir(folder_name):
		if file.endswith(".npz"):
			file_path = os.path.join(folder_name, file)
			data = np.load(file_path)
			
			evaluate(file, data, model, config)






	


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, default="pretrained_models/2022-09-22/16-24-58/ego_model_new.pt")
	parser.add_argument("--evaluation_data_folder", type=str, default="data/kinematic_model_data_val")
	parser.add_argument("--wandb", type=bool, default=False)
	parser.add_argument("--plot_local", type=bool, default=True)
	config = parser.parse_args()

	main(config)




