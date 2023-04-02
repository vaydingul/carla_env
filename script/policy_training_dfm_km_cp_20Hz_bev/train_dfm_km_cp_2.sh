#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"

# Run the world model
python3 train_dfm_km_cp_bev_gt_bev_encoded_policy_fused.py \
	--seed=2023 \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=140 \
	--num_workers=4 \
	--data_path_train="/userfiles/vaydingul20/ground_truth_bev_model_train_data_4_town_02/ground_truth_bev_model_train_data_4_town_02/" \
	--data_path_val="/userfiles/vaydingul20/ground_truth_bev_model_val_data_4_town_02/ground_truth_bev_model_val_data_4_town_02/" \
	--resume=false \
	--num_gpu=8 \
	--master_port="12355" \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--binary_occupancy=false \
	--num_time_step_future=1 \
	--dataset_dilation=1 \
	--debug_render=false \
	--save_interval=5 \
	--input_ego_location=0 \
	--input_ego_yaw=0 \
	--input_ego_speed=1 \
	--delta_target=true \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=256 \
	--num_layer=3 \
	--dropout=0.1 \
	--lane_cost_weight=0.01 \
	--vehicle_cost_weight=0.01 \
	--green_light_cost_weight=-0.01 \
	--yellow_light_cost_weight=0.01 \
	--red_light_cost_weight=0.01 \
	--pedestrian_cost_weight=0.000 \
	--offroad_cost_weight=0.01 \
	--action_mse_weight=10.0 \
	--action_jerk_weight=0.0 \
	--target_progress_weight=-1.0 \
	--target_remainder_weight=1.0 \
	--ego_state_mse_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp-20Hz-bev" \
	--wandb_name="bc+target(bev_encoded_policy_fused)" \
	--ego_forward_model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--world_forward_model_wandb_link="vaydingul/mbl/1gftiw9w" \
	--world_forward_model_checkpoint_number=49
