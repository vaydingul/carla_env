import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_result_eval(
        vehicle_location,
        vehicle_rotation,
        vehicle_velocity,
        vehicle_control,
        elapsed_time,
        location_predicted,
        yaw_predicted,
        speed_predicted,
        savedir):

    plt.figure()
    plt.plot(vehicle_location[:-1, 1],
             vehicle_location[:-1, 0], "r-", label="CARLA")
    plt.plot(location_predicted[:, 1],
             location_predicted[:, 0], "b-", label="NN")
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("Vehicle Trajectory")
    plt.savefig(savedir / "trajectory.png")

    plt.figure()
    plt.plot(elapsed_time[:-1], vehicle_location[:-1, 0], "r-", label="CARLA")
    plt.plot(elapsed_time[:-1], location_predicted[:, 0], "b-", label="NN")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("CARLA vs. NN")
    plt.savefig(savedir / "x-loc.png")

    plt.figure()
    plt.plot(elapsed_time[:-1], vehicle_location[:-1, 1], "r-", label="CARLA")
    plt.plot(elapsed_time[:-1], location_predicted[:, 1], "b-", label="NN")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("CARLA vs. NN")
    plt.savefig(savedir / "y-loc.png")

    plt.figure()
    plt.plot(elapsed_time[:-1],
             np.linalg.norm(vehicle_velocity[:-1],
                            axis=-1),
             "r-",
             label="CARLA")
    plt.plot(elapsed_time[:-1], speed_predicted, "b-", label="NN")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Vehicle Speed")
    plt.savefig(savedir / "speed.png")

    plt.figure()
    plt.plot(elapsed_time, vehicle_control)
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.legend(["Throttle", "Steer", "Brake"])
    plt.title("Control Actions")
    plt.savefig(savedir / "action.png")

    plt.figure()
    plt.plot(elapsed_time[:-
                          1], np.rad2deg(vehicle_rotation[:-
                                                          1, 1]), "r-", label="CARLA")
    plt.plot(elapsed_time[:-1], np.rad2deg(yaw_predicted), "b-", label="NN")
    plt.xlabel("Time")
    plt.ylabel("Yaw")
    plt.title("Vehicle Yaw")
    plt.legend()
    plt.savefig(savedir / "yaw.png")

    plt.close("all")


def plot_result_mpc(state, action, target_state, savedir):
    plt.figure()
    plt.plot(state[:, 0, 1], state[:, 0, 0])
    plt.plot(state[0, 0, 1], state[0, 0, 0], 'go')
    plt.plot(target_state[0, 0, 1], target_state[0, 0, 0], 'ro')
    plt.title("Trajectory")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.savefig(savedir / "trajectory.png")

    plt.figure()
    plt.plot(state[:, 0, 2])
    plt.plot(state.shape[0] - 1, target_state[0, 0, 2], 'ro')
    plt.title("Yaw")
    plt.savefig(savedir / "yaw.png")

    plt.figure()
    plt.plot(state[:, 0, 3])
    plt.plot(state.shape[0] - 1, target_state[0, 0, 3], 'ro')
    plt.title("Speed")
    plt.savefig(savedir / "speed.png")

    plt.figure()
    plt.plot(action[:, 0, 0], label='Throttle')
    plt.plot(action[:, 0, 1], label='Steer')
    plt.legend()
    plt.title("Control Actions")
    plt.savefig(savedir / "action.png")
    plt.close("all")


def plot_result_mpc_path_follow(
        state,
        action,
        vehicle_location,
        vehicle_rotation,
        vehicle_velocity,
        vehicle_control,
        target_state,
        offset,
        end_ix,
        savedir):

    plt.figure()
    plt.plot(state[:, 0, 1], state[:, 0, 0], label="ModelPredictiveControl")
    plt.plot(vehicle_location[offset:end_ix, 1],
             vehicle_location[offset:end_ix, 0], label="Ground Truth")
    plt.plot(target_state[:, 0, 1], target_state[:, 0, 0],
             "ro", label="Intermediate Target Points")
    plt.title("Trajectory")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.legend()
    plt.savefig(savedir / "trajectory.png")

    plt.figure()
    plt.plot(np.linspace(0, 1, action.shape[0]),
             action[:, 0, 0], label='Throttle - ModelPredictiveControl')
    plt.plot(np.linspace(0, 1, action.shape[0]),
             action[:, 0, 1], label='Steer - ModelPredictiveControl')
    plt.plot(np.linspace(0,
                         1,
                         vehicle_control[offset:end_ix].shape[0]),
             vehicle_control[offset:end_ix,
                             0],
             label='Throttle - Ground Truth')
    plt.plot(np.linspace(0,
                         1,
                         vehicle_control[offset:end_ix].shape[0]),
             vehicle_control[offset:end_ix,
                             1],
             label='Steer - Ground Truth')
    plt.legend()
    plt.title("Control Actions")
    plt.savefig(savedir / "action.png")

    plt.figure()
    plt.plot(np.linspace(
        0, 1, state.shape[0]), state[:, 0, 3], label='ModelPredictiveControl')
    plt.plot(np.linspace(0, 1, vehicle_velocity[offset:end_ix].shape[0]), np.linalg.norm(
        vehicle_velocity[offset:end_ix], axis=-1), label='Ground Truth')
    plt.legend()
    plt.title("Speed")
    plt.savefig(savedir / "speed.png")

    plt.figure()
    plt.plot(np.linspace(0, 1, state.shape[0]), np.rad2deg(
        state[:, 0, 2]), label='ModelPredictiveControl')
    plt.plot(np.linspace(0, 1, vehicle_rotation[offset:end_ix].shape[0]), np.rad2deg(
        vehicle_rotation[offset:end_ix, 1]), label='Ground Truth')
    plt.legend()
    plt.title("Yaw")
    plt.savefig(savedir / "yaw.png")

    plt.close("all")


def plot_roc(fpr, tpr, auroc, savedir, multi=False):
    plt.figure(figsize = (5,5))
    lw = 2
    if multi:
        for i in range(fpr.shape[0]):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=lw,
                label=f"ROC curve {i} (AUC = {auroc[i]:.2f})")
    else:
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f"ROC curve (area = {auroc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw,
             linestyle='--', label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(savedir, "roc.png"))
    plt.close("all")


def plot_confusion_matrix(tp, fp, tn, fn, savedir, multi=False):
    plt.figure(figsize = (5,5))
    label = ["True", "False"]
    if multi:
        # Draw a heatmap with the numeric values in each cell
        for i in range(tp.shape[0]):
            cm = np.array([[tp[i], fp[i]], [fn[i], tn[i]]])
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=label, yticklabels=label)
            plt.title(f"Confusion Matrix {i}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(savedir, f"confusion_matrix_{i}.png"))
            plt.close("all")

