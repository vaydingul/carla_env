#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"

# Run the world model
python3 train_dfm_km_cp.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=80 \
	--num_workers=4 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_2/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_2/" \
	--resume=false \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--debug_render=false \
	--save_interval=5 \
	--input_shape_ego_state=4 \
	--action_size=2 \
	--hidden_size=512 \
	--num_layer=6 \
	--lane_cost_weight=0.01 \
	--vehicle_cost_weight=0.01 \
	--green_light_cost_weight=0.000 \
	--yellow_light_cost_weight=0.000 \
	--red_light_cost_weight=0.000 \
	--pedestrian_cost_weight=0.000 \
	--offroad_cost_weight=0.01 \
	--action_mse_weight=1.0 \
	--action_jerk_weight=0.0 \
	--target_mse_weight=0.0 \
	--target_l1_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp-normalized_cost" \
	--wandb_name="vanilla+bc(complex)" \
	--ego_forward_model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--world_forward_model_wandb_link="vaydingul/mbl/r4la61x3" \
	--world_forward_model_checkpoint_number=49