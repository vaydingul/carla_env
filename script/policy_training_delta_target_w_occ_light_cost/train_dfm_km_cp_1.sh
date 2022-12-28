#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"

# Run the world model
python3 train_dfm_km_cp_delta_target_w_occ_ddp.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=50 \
	--num_workers=5 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_4_town_02/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_4_town_02/" \
	--resume=false \
	--num_gpu=2 \
	--master_port="12355" \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--debug_render=false \
	--save_interval=5 \
	--input_ego_location=1 \
	--input_ego_yaw=1 \
	--input_ego_speed=1 \
	--delta_target=true \
	--single_world_state_input=false \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=256 \
	--num_layer=6 \
	--lane_cost_weight=0.1 \
	--vehicle_cost_weight=0.1 \
	--green_light_cost_weight=-0.1 \
	--yellow_light_cost_weight=0.1 \
	--red_light_cost_weight=0.1 \
	--pedestrian_cost_weight=0.000 \
	--offroad_cost_weight=0.1 \
	--action_mse_weight=1.0 \
	--action_jerk_weight=0.0 \
	--target_mse_weight=0.0 \
	--target_l1_weight=0.0 \
	--ego_state_mse_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp" \
	--wandb_name="vanilla+bc(binary_radar)(new_world_model)" \
	--ego_forward_model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--world_forward_model_wandb_link="vaydingul/mbl/2b2rklg9" \
	--world_forward_model_checkpoint_number=49
