import logging
import os 
import argparse


from utilities.model_utils import (
    fetch_checkpoint_from_wandb_run,
    fetch_run_from_wandb_link,
)
from utilities.train_utils import seed_everything, get_device


from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.config_utils import copy_config_to_path, parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config


def main(config):
    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    seed_everything(config["seed"])
    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    pretty_print_config(logger, config)

    # ---------------------------------------------------------------------------- #
    #                                    DEVICE                                    #
    # ---------------------------------------------------------------------------- #
    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN                                  #
    # ---------------------------------------------------------------------------- #
    run = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                    SENSORS                                   #
    # ---------------------------------------------------------------------------- #

    sensor_class_list = sensor_factory(config)
    config["environment"].update({"sensors": sensor_class_list})
    # ---------------------------------------------------------------------------- #
    #                                  ENVIRONMENT                                 #
    # ---------------------------------------------------------------------------- #

    environment_class = environment_factory(config)
    environment = environment_class(config=config["environment"])
    copy_config_to_path(config, os.path.join(environment.renderer_module.save_path, "config.yml")) 

    #                    EGO FORWARD MODEL WANDB RUN CHECKPOINT                    #
    # ---------------------------------------------------------------------------- #

    ego_forward_model_wandb_link = config["wandb_ego_forward_model"]["link"]
    ego_forward_model_checkpoint_number = config["wandb_ego_forward_model"][
        "checkpoint_number"
    ]

    ego_forward_model_run = fetch_run_from_wandb_link(ego_forward_model_wandb_link)
    ego_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
        run=ego_forward_model_run,
        checkpoint_number=ego_forward_model_checkpoint_number,
    )
    ego_forward_model_checkpoint_path = ego_forward_model_checkpoint_object.name

    # Create the model
    ego_forward_model_class = ego_forward_model_factory(ego_forward_model_run.config)
    # Initialize the model
    ego_forward_model = ego_forward_model_class.load_model_from_wandb_run(
        config=ego_forward_model_run.config["ego_forward_model"]["config"],
        checkpoint_path=ego_forward_model_checkpoint_path,
        device=device,
    )

    # ---------------------------------------------------------------------------- #
    #                    WORLD FORWARD MODEL WANDB RUN CHECKPOINT                  #
    # ---------------------------------------------------------------------------- #
    if config["wandb_world_forward_model"] is not None:
        world_forward_model_wandb_link = config["wandb_world_forward_model"]["link"]
        world_forward_model_checkpoint_number = config["wandb_world_forward_model"][
            "checkpoint_number"
        ]

        world_forward_model_run = fetch_run_from_wandb_link(
            world_forward_model_wandb_link
        )
        world_forward_model_checkpoint_object = fetch_checkpoint_from_wandb_run(
            run=world_forward_model_run,
            checkpoint_number=world_forward_model_checkpoint_number,
        )
        world_forward_model_checkpoint_path = world_forward_model_checkpoint_object.name

        # Create the model
        world_forward_model_class = world_forward_model_factory(
            world_forward_model_run.config
        )
        # Initialize the model
        world_forward_model = world_forward_model_class.load_model_from_wandb_run(
            config=world_forward_model_run.config["world_forward_model"]["config"],
            checkpoint_path=world_forward_model_checkpoint_path,
            device=device,
        )

    else:
        world_forward_model = None

    # ---------------------------------------------------------------------------- #
    #                                   COST                                       #
    # ---------------------------------------------------------------------------- #

    cost_class = cost_factory(config)
    cost = cost_class(device, config["cost"]["config"])

    # ---------------------------------------------------------------------------- #
    #                                   OPTIMIZER                                  #
    # ---------------------------------------------------------------------------- #

    optimizer_class = optimizer_factory(config)

    # ---------------------------------------------------------------------------- #
    #                                    TESTER                                    #
    # ---------------------------------------------------------------------------- #
    tester_class = tester_factory(config)
    tester = tester_class(
        environment=environment,
        ego_forward_model=ego_forward_model,
        world_forward_model=world_forward_model,
        cost=cost,
        cost_weight=config["tester"]["cost_weight"],
        device=device,
        optimizer_class=optimizer_class,
        optimizer_config=config["training"]["optimizer"]["config"],
        batch_size=config["tester"]["batch_size"],
        action_size=config["tester"]["action_size"],
        num_optimization_iteration_grad=config["tester"][
            "num_optimization_iteration_grad"
        ],
        num_optimization_iteration_cem=config["tester"][
            "num_optimization_iteration_cem"
        ],
        topk=config["tester"]["topk"],
        init_action=config["tester"]["init_action"],
        init_action_mean=config["tester"]["init_action_mean"],
        init_action_std=config["tester"]["init_action_std"],
        skip_frames=config["tester"]["skip_frames"],
        repeat_frames=config["tester"]["repeat_frames"],
        gradient_clip=config["training"]["gradient_clip"]["enable"],
        gradient_clip_type=config["training"]["gradient_clip"]["type"],
        gradient_clip_value=config["training"]["gradient_clip"]["value"],
        log_video=config["tester"]["log_video"],
        log_video_scale=config["tester"]["log_video_scale"],
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["tester"]["rollout_length"],
        bev_agent_channel=config["tester"]["bev_agent_channel"],
        bev_vehicle_channel=config["tester"]["bev_vehicle_channel"],
        bev_selected_channels=config["tester"]["bev_selected_channels"],
        bev_calculate_offroad=config["tester"]["bev_calculate_offroad"],
    )

    try:
        logger.info("Starting the tester")

        tester.test(run)

        logger.info("Tester finished")

    except Exception as e:
        logger.exception("Tester failed!", exc_info=e)

        logger.info("Closing the environment")

        run.log(
            {
                "INTERRUPTED": True,
                "SUCCESSFUL": False,
                "COLLISION": False,
            }
        )

        run.finish()
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/gradcem/testing/config_extended_bev_20Hz_action_repeat_4_policy_cost_with_pedestrian.yml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = parse_yml(args.config_path)

    config["config_path"] = args.config_path

    main(config)
