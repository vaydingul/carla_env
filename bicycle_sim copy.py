import numpy as np
import matplotlib.pyplot as plt
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
import os 
import pathlib
import torch
DELTA_T = 0.1


def plot_all():
	pass


if __name__ == "__main__":

	fname = "data4_nn.npz"
	data = np.load("data/data4.npz")

	model = KinematicBicycleModel()
	model.state_dict = torch.load("./ego_model_new.pt")
	model.eval()



	vehicle_location = torch.Tensor(data["vehicle_location"])
	vehicle_rotation = torch.Tensor(data["vehicle_rotation"])
	vehicle_velocity = torch.Tensor(data["vehicle_velocity"])
	vehicle_acceleration = torch.Tensor(data["vehicle_acceleration"])
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
	
		

	savedir = pathlib.Path(f"figures/{fname.split('.')[0]}")
	os.makedirs(savedir, exist_ok=True)

	plt.figure()
	plt.plot(vehicle_location[1:, 1], vehicle_location[1:, 0], "r-", label="CARLA")
	plt.plot(location_predicted[:, 1], location_predicted[:, 0], "b-", label="NN")
	#plt.plot(dynamic_bicycle_location[:, 1], dynamic_bicycle_location[:, 0], "g-", label="Dynamic Bicycle")
	plt.legend()
	plt.xlabel("y")
	plt.ylabel("x")
	plt.title("Vehicle Trajectory")
	plt.savefig(savedir / "figure1.png")
	#plt.show()
	print(np.mean(np.abs(vehicle_location[1:, :2] - location_predicted[:])))
	plt.figure()
	plt.plot(elapsed_time[1:], vehicle_location[1:, 0], "r-", label="CARLA")
	plt.plot(elapsed_time[1:], location_predicted[:, 0], "b-", label="NN")
	#plt.plot(elapsed_time, dynamic_bicycle_location[:, 0], "g-", label="Dynamic Bicycle")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("x")
	plt.title("CARLA vs. NN")
	plt.savefig(savedir / "figure2.png")

	plt.figure()
	plt.plot(elapsed_time[1:], vehicle_location[1:, 1], "r-", label="CARLA")
	plt.plot(elapsed_time[1:], location_predicted[:, 1], "b-", label="NN")
	#plt.plot(elapsed_time, dynamic_bicycle_location[:, 1], "g-", label="Dynamic Bicycle")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("y")
	plt.title("CARLA vs. NN")
	plt.savefig(savedir / "figure3.png")

	plt.figure()
	#plt.plot(elapsed_time, np.array([np.linalg.norm(vehicle_velocity[k, :]) for k in range(vehicle_velocity.shape[0])]), "r-", label="CARLA")
	plt.plot(elapsed_time[1:], np.linalg.norm(vehicle_velocity[1:], axis = -1), "r-", label="CARLA")
	plt.plot(elapsed_time[1:], speed_predicted, "b-", label="NN")
	#plt.plot(elapsed_time, dynamic_bicycle_speed, "g-", label="Dynamic Bicycle")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Speed")
	plt.title("Vehicle Speed")
	plt.savefig(savedir / "figure4.png")
	#plt.show()


	plt.figure()
	plt.plot(elapsed_time, vehicle_control)
	plt.xlabel("Time")
	plt.ylabel("Control")
	plt.legend(["Throttle", "Steer", "Brake"])
	plt.title("Control Actions")
	plt.savefig(savedir / "figure7.png")
	#plt.show()


	plt.figure()
	plt.plot(elapsed_time[1:], np.rad2deg(vehicle_rotation[1:, 1]), "r-", label="CARLA")
	plt.plot(elapsed_time[1:], np.rad2deg(yaw_predicted), "b-", label="NN")
	#plt.plot(elapsed_time, np.rad2deg(dynamic_bicycle_yaw), "g-", label="Dynamic Bicycle")
	plt.xlabel("Time")
	plt.ylabel("Yaw")
	plt.title("Vehicle Yaw")
	plt.legend()
	plt.savefig(savedir / "figure8.png")
	#plt.show()





