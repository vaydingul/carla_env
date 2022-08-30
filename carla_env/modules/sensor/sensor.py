from carla_env.modules import module


import carla


class SensorModule(module.Module):
    """Concrete implementation of Module abstract base class for sensor management"""

    def __init__(self, config, client) -> None:
        super().__init__()

        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()

    def _start(self):
        """Start the sensor module"""
        pass
    def _stop(self):
        """Stop the sensor module"""
        pass

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, image):
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

    def _set_default_config(self):
        """Set the default config of the sensor"""
        self.config = {}

    def attach_to_actor(self, actor):
        """Attach the sensor to an actor"""
        actor.sensor_dict[f"{self.__class__.__name__}"] = self
        self.actor = actor
