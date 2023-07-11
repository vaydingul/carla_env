import logging

import numpy as np

from carla_env.modules.sensor import sensor
from enum import IntEnum
from queue import deque
from statistics import mean
from agents.navigation.local_planner import RoadOption


logger = logging.getLogger(__name__)


class SituationState(IntEnum):
    """Enum class for situation state"""

    DRIVE = 1
    STOP = 2
    STEER = 3
    TURN = 4
    JUNCTION = 5



class SituationSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for
    situation sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.steer_deque = deque(maxlen=self.config["steer_deque_maxlen"])

        self.reset()

    def _start(self):
        """Start the sensor module"""
        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""
        self.is_attached = False

    def _tick(self, *args, **kwargs):
        """Tick the sensor"""
        self._get_sensor_data(*args, **kwargs)

    def _get_sensor_data(self, *args, **kwargs):
        """Get the sensor data"""

        data = self._get_required_data()

        self.situation = SituationState.DRIVE

        self.turn_command_available = False
        self.steer_threshold_exceeded = False
        self.steer_mean = 0.0

        if "next_command" in kwargs:
            self.next_command = kwargs["next_command"]
            if self.next_command in [
                RoadOption.LEFT,
                RoadOption.RIGHT,
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ]:
                self.turn_command_available = True

        abs_steer = abs(data["control"]["steer"])
        self.steer_deque.append(abs_steer)

        if len(self.steer_deque) > (self.config["steer_deque_maxlen"] // 2):
            self.steer_mean = mean(self.steer_deque)
            self.steer_threshold_exceeded = (
                self.steer_mean > self.config["steer_threshold"]
            )

        self.is_junction = data["location_waypoint"]["is_junction"]
        self.is_at_traffic_light = data["is_at_traffic_light"]
        self.traffic_light_state = data["traffic_light_state"]

        if self.is_at_traffic_light and (
            self.traffic_light_state == "Red" or self.traffic_light_state == "Yellow"
        ):
            self.situation = SituationState.STOP

        elif self.is_junction:
            self.situation = SituationState.JUNCTION

        elif self.turn_command_available:
            # self.situation = SituationState.TURN
            pass

        elif self.steer_threshold_exceeded:
            # self.situation = SituationState.STEER
            pass

        else:
            self.situation = SituationState.DRIVE

        data = {
            "frame": data["frame"],
            "situation": self.situation,
        }

        if self.save_to_queue:
            self._queue_operation(data)

    def _get_required_data(self):
        actor = self.actor.get_actor()

        transform_ = actor.get_transform()
        transform_forward_vector_ = transform_.get_forward_vector()
        transform_up_vector_ = transform_.get_up_vector()
        transform_right_vector_ = transform_.get_right_vector()
        location_ = actor.get_location()
        rotation_ = actor.get_transform().rotation
        velocity_ = actor.get_velocity()
        angular_velocity_ = actor.get_angular_velocity()
        acceleration_ = actor.get_acceleration()
        control_ = actor.get_control()
        world_ = actor.get_world()

        transform = {
            "matrix": transform_.get_matrix(),
            "inverse_matrix": transform_.get_inverse_matrix(),
            "forward_vector": {
                "x": transform_forward_vector_.x,
                "y": transform_forward_vector_.y,
                "z": transform_forward_vector_.z,
            },
            "up_vector": {
                "x": transform_up_vector_.x,
                "y": transform_up_vector_.y,
                "z": transform_up_vector_.z,
            },
            "right_vector": {
                "x": transform_right_vector_.x,
                "y": transform_right_vector_.y,
                "z": transform_right_vector_.z,
            },
        }

        location = {
            "x": location_.x,
            "y": location_.y,
            "z": location_.z,
        }

        rotation = {
            "roll": rotation_.roll,
            "pitch": rotation_.pitch,
            "yaw": rotation_.yaw,
        }

        velocity = {
            "x": velocity_.x,
            "y": velocity_.y,
            "z": velocity_.z,
        }

        angular_velocity = {
            "x": angular_velocity_.x,
            "y": angular_velocity_.y,
            "z": angular_velocity_.z,
        }

        acceleration = {
            "x": acceleration_.x,
            "y": acceleration_.y,
            "z": acceleration_.z,
        }

        control = {
            "throttle": control_.throttle,
            "steer": control_.steer,
            "brake": control_.brake,
            "hand_brake": control_.hand_brake,
            "reverse": control_.reverse,
            "manual_gear_shift": control_.manual_gear_shift,
            "gear": control_.gear,
        }

        world = world_.get_map().name

        is_at_traffic_light = actor.is_at_traffic_light()
        speed_limit = actor.get_speed_limit()

        traffic_light_state_ = actor.get_traffic_light_state()
        traffic_light_state = str(traffic_light_state_)

        location_waypoint_ = world_.get_map().get_waypoint(location_)
        location_waypoint = {
            "transform": {
                "matrix": location_waypoint_.transform.get_matrix(),
                "inverse_matrix": location_waypoint_.transform.get_inverse_matrix(),
                "forward_vector": {
                    "x": location_waypoint_.transform.get_forward_vector().x,
                    "y": location_waypoint_.transform.get_forward_vector().y,
                    "z": location_waypoint_.transform.get_forward_vector().z,
                },
                "up_vector": {
                    "x": location_waypoint_.transform.get_up_vector().x,
                    "y": location_waypoint_.transform.get_up_vector().y,
                    "z": location_waypoint_.transform.get_up_vector().z,
                },
                "right_vector": {
                    "x": location_waypoint_.transform.get_right_vector().x,
                    "y": location_waypoint_.transform.get_right_vector().y,
                    "z": location_waypoint_.transform.get_right_vector().z,
                },
            },
            "location": {
                "x": location_waypoint_.transform.location.x,
                "y": location_waypoint_.transform.location.y,
                "z": location_waypoint_.transform.location.z,
            },
            "rotation": {
                "roll": location_waypoint_.transform.rotation.roll,
                "pitch": location_waypoint_.transform.rotation.pitch,
                "yaw": location_waypoint_.transform.rotation.yaw,
            },
            "road_id": location_waypoint_.road_id,
            "lane_id": location_waypoint_.lane_id,
            "s": location_waypoint_.s,
            "is_junction": location_waypoint_.is_junction,
            "lane_width": location_waypoint_.lane_width,
        }

        data = {
            "transform_": transform_,
            "location_": location_,
            "rotation_": rotation_,
            "velocity_": velocity_,
            "angular_velocity_": angular_velocity_,
            "acceleration_": acceleration_,
            "control_": control_,
            "world_": world_,
            "traffic_light_state_": traffic_light_state_,
            "location_waypoint_": location_waypoint_,
            "frame": self.world.get_snapshot().frame,
            "transform": transform,
            "location": location,
            "rotation": rotation,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "acceleration": acceleration,
            "control": control,
            "world": world,
            "is_at_traffic_light": is_at_traffic_light,
            "speed_limit": speed_limit,
            "traffic_light_state": traffic_light_state,
            "location_waypoint": location_waypoint,
        }

        return data

    def step(self, *args, **kwargs):
        """Step the sensor"""
        self._tick(*args, **kwargs)

    def reset(self):
        """Reset the sensor"""
        self._stop()
        self._start()

    def render(self):
        """Render the sensor"""
        self.render_dict = {
            "situation": self.situation,
            "is_junction": self.is_junction,
            "is_at_traffic_light": self.is_at_traffic_light,
            "traffic_light_state": self.traffic_light_state,
            "steer_deque": self.steer_deque,
            "steer_threshold": self.config["steer_threshold"],
            "steer_deque_maxlen": self.config["steer_deque_maxlen"],
            "steer_threshold_exceeded": self.steer_threshold_exceeded,
            "steer_mean": self.steer_mean,
            "next_command": self.next_command,
            "turn_command_available": self.turn_command_available,
        }
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
        self.config = {
            "steer_deque_maxlen": 5,
            "steer_threshold": 0.5,
        }

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
