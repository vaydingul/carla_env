#!/usr/bin/env bash

echo "DFM with KM CP Training!"


echo "Conda environment is activated"

# Run the world model
python3 train_dfm_km_cp_delta_target_w_occ.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=9 \
	--num_workers=4 \
	--data_path_train="./data/ground_truth_bev_model_dummy_data/" \
	--data_path_val="./data/ground_truth_bev_model_dummy_data/" \
	--resume=false \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--debug_render=true \
	--save_interval=100 \
	--input_ego_location=1 \
	--input_ego_yaw=1 \
	--input_ego_speed=1 \
	--delta_target=true \
	--single_world_state_input=false \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=16 \
	--num_layer=4 \
	--lane_cost_weight=0.1 \
	--vehicle_cost_weight=0.1 \
	--green_light_cost_weight=-0.1 \
	--yellow_light_cost_weight=0.1 \
	--red_light_cost_weight=0.1 \
	--pedestrian_cost_weight=0.000 \
	--offroad_cost_weight=0.1 \
	--action_mse_weight=0.0 \
	--action_jerk_weight=0.0 \
	--target_mse_weight=0.0 \
	--target_l1_weight=0.0 \
	--ego_state_mse_weight=3.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dummy" \
	--wandb_name="vanilla+ego_state_loss_4" \
	--ego_forward_model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--world_forward_model_wandb_link="vaydingul/mbl/r4la61x3" \
	--world_forward_model_checkpoint_number=49
