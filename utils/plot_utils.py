import matplotlib.pyplot as plt
import numpy as np


def plot_result(vehicle_location, vehicle_rotation, vehicle_velocity, vehicle_control, elapsed_time, location_predicted, yaw_predicted, speed_predicted, savedir):

	print(savedir)
	print(f"Location MAE: {np.mean(np.abs(vehicle_location[:-1, :2] - location_predicted))}")
	print(f"Orientation Cos MAE: {np.mean(np.abs(np.cos(vehicle_rotation[:-1, 1:2]) - np.cos(yaw_predicted)))}")
	print(f"Orientation Sin MAE: {np.mean(np.abs(np.sin(vehicle_rotation[:-1, 1:2]) - np.sin(yaw_predicted)))}")

	plt.figure()
	plt.plot(vehicle_location[:-1, 1], vehicle_location[:-1, 0], "r-", label="CARLA")
	plt.plot(location_predicted[:, 1], location_predicted[:, 0], "b-", label="NN")
	plt.legend()
	plt.xlabel("y")
	plt.ylabel("x")
	plt.title("Vehicle Trajectory")
	plt.savefig(savedir / "trajectory.png")
	

	plt.figure()
	plt.plot(elapsed_time[:-1], vehicle_location[:-1, 0], "r-", label="CARLA")
	plt.plot(elapsed_time[:-1], location_predicted[:, 0], "b-", label="NN")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("x")
	plt.title("CARLA vs. NN")
	plt.savefig(savedir / "x-loc.png")

	plt.figure()
	plt.plot(elapsed_time[:-1], vehicle_location[:-1, 1], "r-", label="CARLA")
	plt.plot(elapsed_time[:-1], location_predicted[:, 1], "b-", label="NN")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("y")
	plt.title("CARLA vs. NN")
	plt.savefig(savedir / "y-loc.png")

	plt.figure()
	plt.plot(elapsed_time[:-1], np.linalg.norm(vehicle_velocity[:-1], axis = -1), "r-", label="CARLA")
	plt.plot(elapsed_time[:-1], speed_predicted, "b-", label="NN")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Speed")
	plt.title("Vehicle Speed")
	plt.savefig(savedir / "speed.png")


	plt.figure()
	plt.plot(elapsed_time, vehicle_control)
	plt.xlabel("Time")
	plt.ylabel("Control")
	plt.legend(["Throttle", "Steer", "Brake"])
	plt.title("Control Actions")
	plt.savefig(savedir / "action.png")


	plt.figure()
	plt.plot(elapsed_time[:-1], np.rad2deg(vehicle_rotation[:-1, 1]), "r-", label="CARLA")
	plt.plot(elapsed_time[:-1], np.rad2deg(yaw_predicted), "b-", label="NN")
	plt.xlabel("Time")
	plt.ylabel("Yaw")
	plt.title("Vehicle Yaw")
	plt.legend()
	plt.savefig(savedir / "yaw.png")
	
	plt.close("all")