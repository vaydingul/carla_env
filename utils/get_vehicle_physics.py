from carla_env import carla_env_basic

import time
import logging
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    c = carla_env_basic.CarlaEnvironment(None)
    vehicle = c.actor.get_actor()
    vehicle_physics_control = vehicle.get_physics_control()
    print(f"Vehicle Mass: {vehicle_physics_control.mass}")
    print(f"Vehicle Center of Mass: {vehicle_physics_control.center_of_mass}")
    print(f"Vehicle Drag Coefficient: {vehicle_physics_control.drag_coefficient}")

    wheels = vehicle_physics_control.wheels
    front_left = wheels[0]
    front_right = wheels[1]
    rear_left = wheels[2]
    rear_right = wheels[3]

    print(f"Front Left Tire Friction: {front_left.tire_friction}")
    print(f"Front Right Tire Friction: {front_right.tire_friction}")
    print(f"Rear Left Tire Friction: {rear_left.tire_friction}")
    print(f"Rear Right Tire Friction: {rear_right.tire_friction}")

    print(f"Front Left Position (x, y, z): {front_left.position}")
    print(f"Front Right Position (x, y, z): {front_right.position}")
    print(f"Rear Left Position (x, y, z): {rear_left.position}")
    print(f"Rear Right Position (x, y, z): {rear_right.position}")

    print(f"Front Left Max Steer Angle: {front_left.max_steer_angle}")
    print(f"Front Right Max Steer Angle: {front_right.max_steer_angle}")
    print(f"Rear Left Max Steer Angle: {rear_left.max_steer_angle}")
    print(f"Rear Right Max Steer Angle: {rear_right.max_steer_angle}")

    print(
        f"Front Left Lateral Stiffness Value (Cornering Stiffness): {front_left.lat_stiff_value}"
    )
    print(
        f"Front Right Lateral Stiffness Value (Cornering Stiffness): {front_right.lat_stiff_value}"
    )
    print(
        f"Rear Left Lateral Stiffness Value (Cornering Stiffness): {rear_left.lat_stiff_value}"
    )
    print(
        f"Rear Right Lateral Stiffness Value (Cornering Stiffness): {rear_right.lat_stiff_value}"
    )

    c.close()
