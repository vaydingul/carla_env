#!/usr/bin/env bash

wandb_links=("vaydingul/mbl/25zll1v7" "vaydingul/mbl/14k60iqj" "vaydingul/mbl/gob4jjru" "vaydingul/mbl/3g4s21ye")

# Run the world model

for i in "${wandb_links[@]}"; do
	wandb_link=$i
	echo $wandb_link
	python3 eval_world_forward_model.py \
		--wandb_link=$wandb_link
done
