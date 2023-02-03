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
    vehicle = c.actor_module.get_actor()
    vehicle_physics_control = vehicle.get_physics_control()

    for _ in range(10):
        c.step()


    # Write information to a file in a folder called docs
    if not os.path.exists("docs"):
        os.makedirs("docs")

    with open("docs/vehicle_physics.txt", "w") as f:

        f.write(f"Vehicle Mass: {vehicle_physics_control.mass}\n")
        f.write(f"Vehicle Center of Mass: {vehicle_physics_control.center_of_mass}\n")
        f.write(
            f"Vehicle Drag Coefficient: {vehicle_physics_control.drag_coefficient}\n"
        )

        f.write(f"Vehicle Location (x, y, z): {vehicle.get_location()}\n")
        f.write(f"Vehicle Rotation (pitch, yaw, roll): {vehicle.get_transform().rotation}\n")
        f.write(f"Vehicle Bounding Box Extent (x, y, z): {vehicle.bounding_box.extent}\n")
        f.write(f"Vehicle Bounding Box Location (x, y, z): {vehicle.bounding_box.location}\n")


        wheels = vehicle_physics_control.wheels
        front_left = wheels[0]
        front_right = wheels[1]
        rear_left = wheels[2]
        rear_right = wheels[3]

        f.write(f"Front Left Tire Friction: {front_left.tire_friction}\n")
        f.write(f"Front Right Tire Friction: {front_right.tire_friction}\n")
        f.write(f"Rear Left Tire Friction: {rear_left.tire_friction}\n")
        f.write(f"Rear Right Tire Friction: {rear_right.tire_friction}\n")

        f.write(f"Front Left Position (x, y, z): {front_left.position}\n")
        f.write(f"Front Right Position (x, y, z): {front_right.position}\n")
        f.write(f"Rear Left Position (x, y, z): {rear_left.position}\n")
        f.write(f"Rear Right Position (x, y, z): {rear_right.position}\n")

        f.write(f"Front Left Max Steer Angle: {front_left.max_steer_angle}\n")
        f.write(f"Front Right Max Steer Angle: {front_right.max_steer_angle}\n")
        f.write(f"Rear Left Max Steer Angle: {rear_left.max_steer_angle}\n")
        f.write(f"Rear Right Max Steer Angle: {rear_right.max_steer_angle}\n")

        f.write(
            f"Front Left Lateral Stiffness Value (Cornering Stiffness): {front_left.lat_stiff_value}\n"
        )
        f.write(
            f"Front Right Lateral Stiffness Value (Cornering Stiffness): {front_right.lat_stiff_value}\n"
        )
        f.write(
            f"Rear Left Lateral Stiffness Value (Cornering Stiffness): {rear_left.lat_stiff_value}\n"
        )
        f.write(
            f"Rear Right Lateral Stiffness Value (Cornering Stiffness): {rear_right.lat_stiff_value}\n"
        )

    c.close()
