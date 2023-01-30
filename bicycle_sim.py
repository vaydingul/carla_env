import numpy as np
import matplotlib.pyplot as plt
from carla_env.models.static.vehicle import KinematicBicycleModel, DynamicBicycleModel
import os
import pathlib

DELTA_T = 0.1


def plot_all():
    pass


if __name__ == "__main__":

    fname = "dynamic_kinematic_model_data_3.npz"
    data = np.load("data/kinematic_model_data_train/dynamic_kinematic_model_data_3.npz")

    vehicle_location = data["vehicle_location"]
    vehicle_rotation = data["vehicle_rotation"]
    vehicle_velocity = data["vehicle_velocity"]
    vehicle_acceleration = data["vehicle_acceleration"]
    vehicle_control = data["vehicle_control"]
    elapsed_time = data["elapsed_time"]

    print(f"Total Elapsed Time: {elapsed_time[-1]}")
    print(f"Number of Steps: {elapsed_time.shape[0]}")

    kinematic_bicycle_model = KinematicBicycleModel(
        x=vehicle_location[0, 0],
        y=vehicle_location[0, 1],
        yaw=vehicle_rotation[0, 1],
        v=np.linalg.norm(vehicle_velocity[0, :]),
    )
    dynamic_bicycle_model = DynamicBicycleModel(
        x=vehicle_location[0, 0],
        y=vehicle_location[0, 1],
        yaw=vehicle_rotation[0, 1],
        velocity_x=vehicle_velocity[0, 0],
        velocity_y=vehicle_velocity[0, 1],
        omega=0,
    )

    kinematic_bicycle_location = np.zeros((elapsed_time.shape[0], 2))
    dynamic_bicycle_location = np.zeros((elapsed_time.shape[0], 2))

    kinematic_bicycle_yaw = np.zeros((elapsed_time.shape[0], 1))
    dynamic_bicycle_yaw = np.zeros((elapsed_time.shape[0], 1))

    kinematic_bicycle_speed = np.zeros((elapsed_time.shape[0]))
    dynamic_bicycle_speed = np.zeros((elapsed_time.shape[0]))

    kinematic_bicycle_acceleration = np.zeros((elapsed_time.shape[0]))
    dynamic_bicycle_acceleration = np.zeros((elapsed_time.shape[0]))

    for k in range(elapsed_time.shape[0]):

        action = vehicle_control[k, :]

        if action[0] > 0 and action[2] == 0.0:
            acceleration = action[0]
        else:
            acceleration = -action[2]

        steer = action[1]

        # kinematic_bicycle_model.step(acceleration, steer, DELTA_T)
        # dynamic_bicycle_model.step(acceleration, steer, DELTA_T)
        kinematic_bicycle_model.step(vehicle_acceleration[k, 0], steer, DELTA_T)
        dynamic_bicycle_model.step(vehicle_acceleration[k, 0], steer, DELTA_T)

        kinematic_bicycle_location[k, :] = (
            kinematic_bicycle_model.x,
            kinematic_bicycle_model.y,
        )
        dynamic_bicycle_location[k, :] = (
            dynamic_bicycle_model.x,
            dynamic_bicycle_model.y,
        )

        kinematic_bicycle_speed[k] = kinematic_bicycle_model.v
        dynamic_bicycle_speed[k] = dynamic_bicycle_model.velocity_x

        # acceleration
        kinematic_bicycle_acceleration[k] = vehicle_acceleration[k, 0]
        # acceleration
        dynamic_bicycle_acceleration[k] = vehicle_acceleration[k, 0]

        kinematic_bicycle_yaw[k] = kinematic_bicycle_model.yaw
        dynamic_bicycle_yaw[k] = dynamic_bicycle_model.yaw

    savedir = pathlib.Path(f"figures/{fname.split('.')[0]}")
    os.makedirs(savedir, exist_ok=True)

    plt.figure()
    plt.plot(vehicle_location[:, 1], vehicle_location[:, 0], "r-", label="CARLA")
    plt.plot(
        kinematic_bicycle_location[:, 1],
        kinematic_bicycle_location[:, 0],
        "b-o",
        label="Kinematic Bicycle",
    )
    # plt.plot(dynamic_bicycle_location[:, 1], dynamic_bicycle_location[:, 0], "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("CARLA vs. Bicycle Model")
    plt.savefig(savedir / "figure1.png")
    # plt.show()

    plt.figure()
    plt.plot(elapsed_time, vehicle_location[:, 0], "r-", label="CARLA")
    plt.plot(
        elapsed_time, kinematic_bicycle_location[:, 0], "b-", label="Kinematic Bicycle"
    )
    # plt.plot(elapsed_time, dynamic_bicycle_location[:, 0], "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("CARLA vs. Bicycle Model")
    plt.savefig(savedir / "figure2.png")

    plt.figure()
    plt.plot(elapsed_time, vehicle_location[:, 1], "r-", label="CARLA")
    plt.plot(
        elapsed_time, kinematic_bicycle_location[:, 1], "b-", label="Kinematic Bicycle"
    )
    # plt.plot(elapsed_time, dynamic_bicycle_location[:, 1], "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("CARLA vs. Bicycle Model")
    plt.savefig(savedir / "figure3.png")

    plt.figure()
    # plt.plot(elapsed_time, np.array([np.linalg.norm(vehicle_velocity[k, :]) for k in range(vehicle_velocity.shape[0])]), "r-", label="CARLA")
    plt.plot(elapsed_time, vehicle_velocity[:, 0], "r-", label="CARLA")
    plt.plot(elapsed_time, kinematic_bicycle_speed, "b-", label="Kinematic Bicycle")
    # plt.plot(elapsed_time, dynamic_bicycle_speed, "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("CARLA vs. Bicycle Model")
    plt.savefig(savedir / "figure4.png")
    # plt.show()

    plt.figure()
    # plt.plot(elapsed_time, np.array([np.linalg.norm(vehicle_acceleration[k, :]) for k in range(vehicle_acceleration.shape[0])]), "r-", label="CARLA")
    plt.plot(elapsed_time, vehicle_acceleration[:, 0], "r-", label="CARLA")
    # plt.plot(elapsed_time, kinematic_bicycle_acceleration, "b-", label="Kinematic Bicycle")
    # plt.plot(elapsed_time, dynamic_bicycle_acceleration, "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Acceleration-x")
    plt.title("Vehicle Acceleration Response in CARLA")
    plt.savefig(savedir / "figure5.png")
    # plt.show()

    plt.figure()
    # plt.plot(elapsed_time, np.array([np.linalg.norm(vehicle_acceleration[k, :]) for k in range(vehicle_acceleration.shape[0])]), "r-", label="CARLA")
    plt.plot(elapsed_time, vehicle_acceleration[:, 1], "r-", label="CARLA")
    plt.plot(
        elapsed_time, kinematic_bicycle_acceleration, "b-", label="Kinematic Bicycle"
    )
    # plt.plot(elapsed_time, dynamic_bicycle_acceleration, "g-", label="Dynamic Bicycle")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Acceleration-y")
    plt.title("Vehicle Acceleration Response in CARLA")
    plt.savefig(savedir / "figure6.png")
    # plt.show()

    plt.figure()
    plt.plot(elapsed_time, vehicle_control)
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.legend(["Throttle", "Steer", "Brake"])
    plt.title("Control Actions")
    plt.savefig(savedir / "figure7.png")
    # plt.show()

    plt.figure()
    plt.plot(elapsed_time, np.rad2deg(vehicle_rotation[:, 1]), "r-", label="CARLA")
    plt.plot(
        elapsed_time, np.rad2deg(kinematic_bicycle_yaw), "b-", label="Kinematic Bicycle"
    )
    # plt.plot(elapsed_time, np.rad2deg(dynamic_bicycle_yaw), "g-", label="Dynamic Bicycle")
    plt.xlabel("Time")
    plt.ylabel("Yaw")
    plt.title("Vehicle Yaw")
    plt.legend()
    plt.savefig(savedir / "figure8.png")
    # plt.show()
