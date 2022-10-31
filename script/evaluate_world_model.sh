#!/usr/bin/env bash

#wandb_links=("vaydingul/mbl/25zll1v7" "vaydingul/mbl/14k60iqj" "vaydingul/mbl/gob4jjru" "vaydingul/mbl/3g4s21ye" "vaydingul/mbl/1gi2qjiw")
wandb_links=("vaydingul/mbl/phys7134" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/203kw46a" "vaydingul/mbl/3aqhglkb")
checkpoint_numbers=(9 14 19 49)
length=${#wandb_links[@]}
# Run the world model

for ((i = 0; i < $length; i++)); do
	echo "${wandb_links[i]}"
	python3 eval_world_forward_model.py \
		--wandb_link="${wandb_links[i]}" \
		--checkpoint_number="${checkpoint_numbers[i]}"
done
