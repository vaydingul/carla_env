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

    while not c.is_done:

        c.step()

    c.close()

    # os.makedirs("images", exist_ok=True)
    # for ix, rgb_image in enumerate(rgbs):

    #     img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    #     cv2.imwrite("images/{}.png".format(ix), img)

    vehicle_location_vehicle_frame_list = []
    vehicle_velocity_vehicle_frame_list = []
    vehicle_acceleration_vehicle_frame_list = []
    vehicle_control = []
    snapshot_list = []
    for _ in range(c.data.qsize()):

        data_point = c.data.get()
        snapshot_list.append(data_point["snapshot"])

        if data_point != {} and "VehicleSensorModule" in data_point.keys():

            vehicle_control.append(
                data_point["VehicleSensorModule"]["control"])

            vehicle_location = data_point["VehicleSensorModule"]["location"]
            vehicle_velocity = data_point["VehicleSensorModule"]["velocity"]
            vehicle_acceleration = data_point["VehicleSensorModule"]["acceleration"]

            vehicle_location = np.array(
                [vehicle_location.x, vehicle_location.y, vehicle_location.z, 1])
            vehicle_velocity = np.array(
                [vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z, 1])
            vehicle_acceleration = np.array(
                [vehicle_acceleration.x, vehicle_acceleration.y, vehicle_acceleration.z, 1])

            vehicle_location_vehicle_frame = c.initial_vehicle_transform.get_inverse_matrix() @ vehicle_location
            vehicle_velocity_vehicle_frame = vehicle_velocity #c.initial_vehicle_transform.get_inverse_matrix() @ vehicle_velocity
            vehicle_acceleration_vehicle_frame = vehicle_acceleration #c.initial_vehicle_transform.get_inverse_matrix(
            #) @ vehicle_acceleration

            vehicle_location_vehicle_frame_list.append(
                vehicle_location_vehicle_frame[:-1])
            vehicle_velocity_vehicle_frame_list.append(
                vehicle_velocity_vehicle_frame[:-1])
            vehicle_acceleration_vehicle_frame_list.append(
                vehicle_acceleration_vehicle_frame[:-1])

    vehicle_location_vehicle_frame_list = np.array(
        vehicle_location_vehicle_frame_list)
    vehicle_control = np.array(vehicle_control)
    elapsed_time = np.array(
        [snapshot.timestamp.elapsed_seconds for snapshot in snapshot_list])

    np.savez("data2", vehicle_location=vehicle_location_vehicle_frame_list, vehicle_velocity=vehicle_velocity_vehicle_frame_list,
             vehicle_acceleration=vehicle_acceleration_vehicle_frame_list, vehicle_control=vehicle_control, elapsed_time=elapsed_time)

    exit()

    plt.figure()
    plt.plot(vehicle_location_vehicle_frame_list[:, 0])
    plt.figure()
    plt.plot(vehicle_location_vehicle_frame_list[:, 1])
    plt.figure()
    plt.plot(vehicle_location_vehicle_frame_list[:, 2])
    plt.figure()
    plt.plot(vehicle_control)
    plt.show()
