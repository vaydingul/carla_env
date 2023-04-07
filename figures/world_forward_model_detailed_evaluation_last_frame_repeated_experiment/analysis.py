import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CLASSES_SELECTED_MODEL = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASSES_SELECTED_LAST_FRAME_REPEATED = [0, 1, 2, 3, 4, 5, 6, 10, 11]
CLASS_NAME = [
    "ROAD",
    "ROAD ON",
    "ROAD OFF",
    "ROAD RED YELLOW",
    "ROAD GREEN",
    "LANES",
    "VEHICLES",
    "PEDESTRIANS",
    "OFFROAD",
]

# Different colors for each class
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def main(path):

    assert len(CLASSES_SELECTED_MODEL) == len(
        CLASSES_SELECTED_LAST_FRAME_REPEATED
    ), "The number of classes selected for the model and the last frame repeated should be the same"

    last_frame_repeated_results = []
    model_results = []
    num_time_steps = []
    # Load data

    # Traverse all the folders
    for folder in os.listdir(path):
        if folder.startswith("num_predict"):
            # Go into the folder
            os.chdir(os.path.join(path, folder))
            # Load the data
            last_frame_repeated_results.append(pd.read_csv("last_frame/metrics.csv"))
            model_results.append(pd.read_csv("model/metrics.csv"))
            num_time_steps.append(int(folder.split("_")[2]))
            # Go back to the parent folder
            os.chdir("..")

    # Plot the results
    # Create (10, 10) figure
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.flatten()
    # For each class
    for i in range(len(CLASSES_SELECTED_MODEL)):
        iou_list_model = []
        iou_list_last_frame_repeated = []
        # For each time step

        for j in range(len(num_time_steps)):
            iou_list_model.append(
                [
                    num_time_steps[j],
                    model_results[j].iloc[CLASSES_SELECTED_MODEL[i]]["iou"],
                ]
            )
            iou_list_last_frame_repeated.append(
                [
                    num_time_steps[j],
                    last_frame_repeated_results[j].iloc[
                        CLASSES_SELECTED_LAST_FRAME_REPEATED[i]
                    ]["iou"],
                ]
            )

        # Sort based on the first element
        iou_list_model.sort(key=lambda x: x[0])
        iou_list_last_frame_repeated.sort(key=lambda x: x[0])

        # Create a numpy array for iou lists
        iou_list_model = np.array(iou_list_model)
        iou_list_last_frame_repeated = np.array(iou_list_last_frame_repeated)

        axs[i].plot(
            iou_list_model[:, 0],
            iou_list_model[:, 1],
            label=f"Model",
            color=COLORS[i],
            ls="--",
        )
        axs[i].plot(
            iou_list_last_frame_repeated[:, 0],
            iou_list_last_frame_repeated[:, 1],
            label=f"Last Frame Repeated",
            color=COLORS[i],
        )

        axs[i].set_xlabel("Number of Time Steps")
        axs[i].set_ylabel("IoU")
        axs[i].set_title(f"{CLASS_NAME[i]}")
        axs[i].legend()

    fig.tight_layout()
    plt.savefig(f"{path}/iou_comparison.png")
    plt.show()


if __name__ == "__main__":

    path = "/home/volkan/Documents/Codes/carla_env/figures/world_forward_model_detailed_evaluation_last_frame_repeated_experiment"

    main(path)
