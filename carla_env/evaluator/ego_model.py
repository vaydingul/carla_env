import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class Evaluator(object):

	def __init__(
			self,
			model,
			dataloader,
			device,
		sequence_length=10,
			save_path=None):
		self.model = model
		self.dataloader = dataloader
		self.device = device
		self.sequence_length = sequence_length
		self.save_path = save_path

		# Create folder at save_path
		if self.save_path is not None:
			if not os.path.exists(self.save_path):
				os.makedirs(self.save_path)

		self.model.to(self.device)

	def evaluate(self, render=True, save=True):

		self.model.eval()

		for i, (data) in enumerate(self.dataloader):

			ego_previous_location = data["ego"]["location_array"][:, 0:1, 0:2].to(
				self.device)
			ego_future_location = data["ego"]["location_array"][:, 1:, 0:2].to(
				self.device)
			ego_future_location_predicted_list = []

			ego_previous_yaw = torch.deg2rad(
				data["ego"]["rotation_array"][:, 0:1, 2:].to(self.device))
			ego_future_yaw = torch.deg2rad(
				data["ego"]["rotation_array"][:, 1:, 2:].to(self.device))
			ego_future_yaw_predicted_list = []

			ego_previous_speed = data["ego"]["velocity_array"][:, 0:1].norm(
				2, -1, keepdim=True).to(self.device)
			ego_future_speed = data["ego"]["velocity_array"][:, 1:].norm(
				2, -1, keepdim=True).to(self.device)
			ego_future_speed_predicted_list = []

			ego_action = data["ego"]["control_array"].to(self.device)
			ego_action[..., 0] -= ego_action[..., -1]

			ego_state = {"location": ego_previous_location,
						 "yaw": ego_previous_yaw,
						 "speed": ego_previous_speed}

			for k in range(self.sequence_length - 1):
				
				ego_state_next = self.model(ego_state, ego_action[:, k:k+1])
				
				ego_future_location_predicted_list.append(
					ego_state_next["location"])
				ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
				ego_future_speed_predicted_list.append(ego_state_next["speed"])

				ego_state = ego_state_next

			ego_future_location_predicted = torch.cat(
				ego_future_location_predicted_list, 1)
			ego_future_yaw_predicted = torch.cat(
				ego_future_yaw_predicted_list, 1)
			ego_future_speed_predicted = torch.cat(
				ego_future_speed_predicted_list, 1)


			

			self.plot(ego_future_location, ego_future_location_predicted, ego_future_yaw, ego_future_yaw_predicted, ego_future_speed, ego_future_speed_predicted, ego_action[..., :2], i)

			
			
	def plot(self, ego_future_location, ego_future_location_predicted, ego_future_yaw, ego_future_yaw_predicted, ego_future_speed, ego_future_speed_predicted, ego_action, i):

		ego_future_location = ego_future_location.detach().cpu().numpy().reshape(-1, 2)
		ego_future_location_predicted = ego_future_location_predicted.detach().cpu().numpy().reshape(-1, 2)
		ego_future_yaw = ego_future_yaw.detach().cpu().numpy().reshape(-1, 1)
		ego_future_yaw_predicted = ego_future_yaw_predicted.detach().cpu().numpy().reshape(-1, 1)
		ego_future_speed = ego_future_speed.detach().cpu().numpy().reshape(-1, 1)
		ego_future_speed_predicted = ego_future_speed_predicted.detach().cpu().numpy().reshape(-1, 1)
		ego_action = ego_action.detach().cpu().numpy().reshape(-1, 2)

		fig, axs = plt.subplots(4, 1, figsize=(10, 10))
		axs[0].plot(ego_future_location[:, 1], ego_future_location[:, 0], label="Ground Truth")
		axs[0].plot(ego_future_location_predicted[:, 1], ego_future_location_predicted[:, 0], label="Predicted")
		axs[0].set_title("Location")
		axs[0].legend()

		axs[1].plot(ego_future_yaw, label="Ground Truth")
		axs[1].plot(ego_future_yaw_predicted, label="Predicted")
		axs[1].set_title("Yaw")
		axs[1].legend()

		axs[2].plot(ego_future_speed, label="Ground Truth")
		axs[2].plot(ego_future_speed_predicted, label="Predicted")
		axs[2].set_title("Speed")
		axs[2].legend()

		axs[3].plot(ego_action[:, 0], label="Acceleration")
		axs[3].plot(ego_action[:, 1], label="Steer")
		axs[3].set_title("Action")
		axs[3].legend()


		plt.tight_layout()

		if self.save_path is not None:
			plt.savefig(os.path.join(self.save_path, f"plot_{i}.png"))
			plt.close()
