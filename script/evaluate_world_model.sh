#!/usr/bin/env bash

#wandb_links=("vaydingul/mbl/25zll1v7" "vaydingul/mbl/14k60iqj" "vaydingul/mbl/gob4jjru" "vaydingul/mbl/3g4s21ye" "vaydingul/mbl/1gi2qjiw")
#wandb_links=("vaydingul/mbl/3vbs6lik" "vaydingul/mbl/phys7134" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/203kw46a")
wandb_links=("vaydingul/mbl/wvdt0p1k" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/3qkyatvq")

#checkpoint_numbers=(9 -1 -1 9)
checkpoint_numbers=(-1 -1 -1 -1)

length=${#wandb_links[@]}
# Run the world model

for ((i = 0; i < $length; i++)); do
	echo "${wandb_links[i]}"
	python3 eval_world_forward_model.py \
		--wandb_link="${wandb_links[i]}" \
		--checkpoint_number="${checkpoint_numbers[i]}" \
		--data_path_test="data/ground_truth_bev_model_test_data_2/" \
		--test_set_step=10 \
		--threshold=0.5
done
