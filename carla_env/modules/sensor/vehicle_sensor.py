from carla_env.modules.sensor import sensor
from queue import Queue, Empty


class VehicleSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for vehicle sensor management"""

    def __init__(self, config, client, actor=None) -> None:
        super().__init__(config, client)

        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        if actor is not None:
            self.attach_to_actor(actor)

        self.reset()

    def _start(self):
        """Start the sensor module"""
        self.queue = Queue()

    def _stop(self):
        """Stop the sensor module"""
        pass

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self):
        """Get the sensor data"""
        data = {'transform': self.actor.player.get_transform(),
                'location': self.actor.player.get_location(),
                'velocity': self.actor.player.get_velocity(),
                'control': self.actor.last_applied_control,
                'frame': self.world.get_snapshot().frame
                }

        

        self._queue_operation(data)

    def step(self):
        """Step the sensor"""
        self._tick()
        self._get_sensor_data()

    def reset(self):
        """Reset the sensor"""
        self._start()

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

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
