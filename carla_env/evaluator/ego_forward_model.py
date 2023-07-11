import torch
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt

from utilities.train_utils import to, clone, cat, stack, apply_torch_func


class Evaluator(object):
    def __init__(
        self, model, dataloader, device, metric, sequence_length=10, save_path=None
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.metric = metric
        self.sequence_length = sequence_length
        self.save_path = save_path

        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, run, table):
        self.model.eval()

        for i, (data) in enumerate(self.dataloader):
            ego_state_previous = to(
            data["ego"],
            self.device,
            index_end=1,
        )
            ego_state_previous["rotation"] = apply_torch_func(ego_state_previous["rotation"], torch.deg2rad)

            ego_state_future = to(
                data["ego"],
                self.device,
                index_start=1,
                
            )
            ego_state_future["rotation"] = apply_torch_func(ego_state_future["rotation"], torch.deg2rad)

            ego_state_future_predicted_list = []

            ego_action = torch.stack([data["ego"]["control"]["throttle"], 
                                    data["ego"]["control"]["steer"],
                                    data["ego"]["control"]["brake"]], dim=-1).to(self.device)

            ego_action[..., 0] -= ego_action[..., -1]
            ego_action = ego_action[..., :-1]

            for t in range(self.sequence_length - 1):
                control_ = ego_action[:, t]

                ego_state_next = self.model(ego_state_previous, control_)
                ego_state_future_predicted_list.append(ego_state_next)
                ego_state_previous = ego_state_next

            ego_state_future_predicted = cat(ego_state_future_predicted_list, dim=1)

            ego_future_location = ego_state_future["location"]
            ego_future_location_predicted = ego_state_future_predicted["location"]

            ego_future_yaw = ego_state_future["rotation"]["yaw"]
            ego_future_yaw_predicted = ego_state_future_predicted["rotation"]["yaw"]

            ego_future_velocity = ego_state_future["velocity"]
            ego_future_velocity_predicted = ego_state_future_predicted["velocity"]

            ego_future_speed = torch.sqrt(
                torch.square(ego_future_velocity["x"])
                + torch.square(ego_future_velocity["y"])
            )

            ego_future_speed_predicted = torch.sqrt(
                torch.square(ego_future_velocity_predicted["x"])
                + torch.square(ego_future_velocity_predicted["y"])
            )

            metric_location_x = self.metric(
                ego_future_location["x"], ego_future_location_predicted["x"]
            )
            metric_location_y = self.metric(
                ego_future_location["y"], ego_future_location_predicted["y"]
            )

            metric_location = (metric_location_x + metric_location_y) / 2

            metric_rotation = self.metric(
                torch.cos(ego_future_yaw), torch.cos(ego_future_yaw_predicted)
            )
            metric_rotation += self.metric(
                torch.sin(ego_future_yaw), torch.sin(ego_future_yaw_predicted)
            )

            metric_speed = self.metric(ego_future_speed, ego_future_speed_predicted)


            loss = metric_location + metric_rotation

            save_path_plot = self.plot(
                ego_future_location,
                ego_future_location_predicted,
                ego_future_yaw,
                ego_future_yaw_predicted,
                ego_future_velocity,
                ego_future_velocity_predicted,
                ego_action,
                i,
            )

            image = wandb.Image(save_path_plot)
            table.add_data(i, metric_location, metric_rotation, metric_speed, image)

    def plot(
        self,
        ego_future_location,
        ego_future_location_predicted,
        ego_future_yaw,
        ego_future_yaw_predicted,
        ego_future_velocity,
        ego_future_velocity_predicted,
        ego_action,
        i,
    ):
        ego_future_location_x = ego_future_location["x"].detach().cpu().numpy().reshape(-1, 1)
        ego_future_location_predicted_x = (
            ego_future_location_predicted["x"].detach().cpu().numpy().reshape(-1, 1)
        )
        ego_future_location_y = ego_future_location["y"].detach().cpu().numpy().reshape(-1, 1)
        ego_future_location_predicted_y = (
            ego_future_location_predicted["y"].detach().cpu().numpy().reshape(-1, 1)
        )

        # Concatenate x and y in numpy
        ego_future_location = np.concatenate(
            [ego_future_location_x, ego_future_location_y], axis=-1
        )
        ego_future_location_predicted = np.concatenate(
            [ego_future_location_predicted_x, ego_future_location_predicted_y], axis=-1
        )

        ego_future_yaw = ego_future_yaw.detach().cpu().numpy().reshape(-1, 1)
        ego_future_yaw_predicted = (
            ego_future_yaw_predicted.detach().cpu().numpy().reshape(-1, 1)
        )

        ego_future_velocity_x = ego_future_velocity["x"].detach().cpu().numpy().reshape(-1, 1)
        ego_future_velocity_predicted_x = (
            ego_future_velocity_predicted["x"].detach().cpu().numpy().reshape(-1, 1)
        )
        ego_future_velocity_y = ego_future_velocity["y"].detach().cpu().numpy().reshape(-1, 1)
        ego_future_velocity_predicted_y = (
            ego_future_velocity_predicted["y"].detach().cpu().numpy().reshape(-1, 1)
        )

        ego_future_speed = np.sqrt(
            np.square(ego_future_velocity_x) + np.square(ego_future_velocity_y)
        )
        ego_future_speed_predicted = np.sqrt(
            np.square(ego_future_velocity_predicted_x)
            + np.square(ego_future_velocity_predicted_y)
        )

        ego_action = ego_action.detach().cpu().numpy().reshape(-1, 2)

        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        axs[0][0].plot(ego_future_location_x, label="Ground Truth")
        axs[0][0].plot(ego_future_location_predicted_x, label="Predicted")
        # Calculate mean absolute error and add to title
        mean_absolute_error = np.mean(
            np.abs(ego_future_location_x - ego_future_location_predicted_x)
        )
        axs[0][0].set_title(f"Location-X, MAE: {mean_absolute_error:.2f}")
        axs[0][0].legend()

        axs[1][0].plot(ego_future_location_y, label="Ground Truth")
        axs[1][0].plot(ego_future_location_predicted_y, label="Predicted")
        # Calculate mean absolute error and add to title
        mean_absolute_error = np.mean(
            np.abs(ego_future_location_y - ego_future_location_predicted_y)
        )
        axs[1][0].set_title(f"Location-Y, MAE: {mean_absolute_error:.2f}")
        axs[1][0].legend()

        axs[2][0].plot(ego_future_yaw, label="Ground Truth")
        axs[2][0].plot(ego_future_yaw_predicted, label="Predicted")
        # Calculate mean absolute error and add to title
        mean_absolute_error = np.mean(
            np.abs(np.cos(ego_future_yaw) - np.cos(ego_future_yaw_predicted))
        )
        mean_absolute_error += np.mean(
            np.abs(np.sin(ego_future_yaw) - np.sin(ego_future_yaw_predicted))
        )
        axs[2][0].set_title(f"Yaw, MAE: {mean_absolute_error:.2f}")
        axs[2][0].legend()

        axs[0][1].plot(
            ego_future_location_y, ego_future_location_x, label="Ground Truth"
        )
        axs[0][1].plot(
            ego_future_location_predicted_y,
            ego_future_location_predicted_x,
            label="Predicted",
        )
        # Calculate mean absolute error and add to title
        mean_absolute_error = np.mean(
            np.abs(ego_future_location - ego_future_location_predicted)
        )
        axs[0][1].set_title(f"Trajectory, MAE: {mean_absolute_error:.2f}")
        axs[0][1].legend()

        axs[1][1].plot(ego_future_speed, label="Ground Truth")
        axs[1][1].plot(ego_future_speed_predicted, label="Predicted")
        # Calculate mean absolute error and add to title
        mean_absolute_error = np.mean(
            np.abs(ego_future_speed - ego_future_speed_predicted)
        )
        axs[1][1].set_title(f"Speed, MAE: {mean_absolute_error:.2f}")
        axs[1][1].legend()

        axs[2][1].plot(ego_action[:, 0], label="Acceleration")
        axs[2][1].plot(ego_action[:, 1], label="Steer")
        axs[2][1].set_title("Action")
        axs[2][1].legend()

        plt.tight_layout()

        if self.save_path is not None:
            save_path_plot = os.path.join(self.save_path, f"plot_{i}.png")
            plt.savefig(save_path_plot)
            plt.close()

        return save_path_plot
