from carla_env import carla_env

import time



if __name__ == "__main__":

	c = carla_env.CarlaEnvironment(None)

	t_init = time.time()
	
	while True:
		action = [1.0, 0.0, 0.0]
		c.step(action)
		if c.vehicle_sensor.queue.qsize() > 0:
			print(c.vehicle_sensor.queue.get())
		
		#time.sleep(0.01)