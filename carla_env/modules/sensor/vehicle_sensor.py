from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class VehicleSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for vehicle sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""
        pass

    def _stop(self):
        """Stop the sensor module"""
        pass

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self):
        """Get the sensor data"""
        vehicle_control = self.actor.get_actor().get_control()

        vehicle_control_ = [vehicle_control.throttle,
                            vehicle_control.steer, vehicle_control.brake]
        data = {'transform': self.actor.get_actor().get_transform(),
                'location': self.actor.get_actor().get_location(),
                'rotation': self.actor.get_actor().get_transform().rotation,
                'velocity': self.actor.get_actor().get_velocity(),
                'acceleration': self.actor.get_actor().get_acceleration(),
                'control': vehicle_control_,
                'frame': self.world.get_snapshot().frame
                }

        logger.debug(f"Location: {data['location']}")
        logger.debug(f"Rotation: {data['rotation']}")
        logger.debug(f"Velocity: {data['velocity']}")
        logger.debug(f"Acceleration: {data['acceleration']}")
        logger.debug(f"Control: {data['control']}")
        logger.debug(f"Frame: {data['frame']}")

        if self.save_to_queue:
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
        return self.render_dict

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
