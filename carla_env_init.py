from carla_env import carla_env

import time
import logging
import cv2
import os
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

	c = carla_env.CarlaEnvironment(None)

	t_init = time.time()
	rgbs = []
	while time.time()-t_init < 50:
		action = [1.0, 0.0, 0.0]
		c.step(action)
		if c.rgb_sensor.queue.qsize() > 0:
			data = c.rgb_sensor.queue.get()
			rgbs.append(data["data"])
	c.close()


	os.makedirs("images", exist_ok=True)
	for ix, rgb_image in enumerate(rgbs):

		img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			
		cv2.imwrite("images/{}.png".format(ix), img)

		#time.sleep(0.01)