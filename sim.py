from carla_env import carla_env

import time
import logging
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

	c = carla_env.CarlaEnvironment(None)

	t_init = time.time()

	while True:
		
		t_ = time.time()

		if t_ - t_init < 10:

			action = [1.0, 0.0, 0.0]

		elif t_ - t_init > 10 and t_ - t_init < 11:

			action = [0.0, 0.5, 0.0]

		elif t_ - t_init > 11 and t_ - t_init < 12:

			action = [0.0, -0.5, 0.0]

		elif t_ - t_init > 12 and t_ - t_init < 20:

			action = [0.0, 0.0, 1.0]

		else:

			break

		c.step(action)

	c.close()

	# os.makedirs("images", exist_ok=True)
	# for ix, rgb_image in enumerate(rgbs):

	#     img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

	#     cv2.imwrite("images/{}.png".format(ix), img)

	vehicle_location_vehicle_frame_list = []
	vehicle_control = []
	snapshot_list = []
	for _ in range(c.data.qsize()):

		data_point = c.data.get()
		snapshot_list.append(data_point["snapshot"])
		
		if data_point != {} and "VehicleSensorModule" in data_point.keys():

			vehicle_control.append(data_point["VehicleSensorModule"]["control"])

			vehicle_location = data_point["VehicleSensorModule"]["location"]
			vehicle_location = np.array(
				[vehicle_location.x, vehicle_location.y, vehicle_location.z, 1])
			
			vehicle_location_vehicle_frame = c.initial_vehicle_transform.get_inverse_matrix() @ vehicle_location
			vehicle_location_vehicle_frame_list.append(
				vehicle_location_vehicle_frame[:-1])

	vehicle_location_vehicle_frame_list = np.array(
		vehicle_location_vehicle_frame_list)
	vehicle_control = np.array(vehicle_control)

	plt.figure()
	plt.plot(vehicle_location_vehicle_frame_list[:, 0])
	plt.figure()
	plt.plot(vehicle_location_vehicle_frame_list[:, 1])
	plt.figure()
	plt.plot(vehicle_location_vehicle_frame_list[:, 2])
	plt.figure()
	plt.plot(vehicle_control)
	plt.show()
