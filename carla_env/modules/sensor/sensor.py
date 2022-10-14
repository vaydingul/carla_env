from carla_env.modules import module
from queue import Queue


class SensorModule(module.Module):
    """Concrete implementation of Module abstract base class for sensor management"""

    def __init__(self, config, client) -> None:
        super().__init__()

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.render_dict = {}
        self.queue = Queue()

        self.save_to_queue = True

    def _start(self):
        """Start the sensor module"""
        pass

    def _stop(self):
        """Stop the sensor module"""
        pass

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, _):
        """Get the sensor data"""
        pass

    def step(self):
        """Step the sensor"""
        self._tick()

    def reset(self):
        """Reset the sensor"""
        pass

    def render(self):
        """Render the sensor"""
        pass

    def close(self):
        """Close the sensor"""
        pass

    def seed(self):
        """Seed the sensor"""
        pass

    def get_config(self):
        """Get the config of the sensor"""
        return self.config

    def get_queue(self):
        """Get the queue of the sensor"""
        return self.queue

    def _set_default_config(self):
        """Set the default config of the sensor"""
        self.config = {}

    def attach_to_actor(self, actor, id=None):
        """Attach the sensor to an actor"""

        if (id is not None):

            if (id not in actor.sensor_dict.keys()):
                actor.sensor_dict[id] = self
            else:
                raise ("A sensor with same ID exists!")

        else:

            if f"{self.__class__.__name__}_0" not in actor.sensor_dict.keys():

                actor.sensor_dict[f"{self.__class__.__name__}_0"] = self

            else:

                sorted_keys = sorted([int(key_.split("_")[-1])
                                      for key_ in actor.sensor_dict.keys()
                                      if self.__class__.__name__ in key_])

                actor.sensor_dict[f"{self.__class__.__name__}_{sorted_keys[-1] + 1}"] = self

        self.actor = actor
