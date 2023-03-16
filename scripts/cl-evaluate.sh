#!/bin/bash

# Run `./evaluate.sh` on CodaLab.
# Feel free to modify this script.

set -x

dataset_name=openwebtext_wordlength
eval_data=wordlength_eval_data
seed=1111

# Evaluate the GPT-2 model
run_name="eval_${dataset_name}_seed${seed}_step0"
cl run \
  --name $run_name \
  --request-docker-image sangxie513/cs324-p2-codalab-gpu \
  --request-memory 32g \
  :src \
  :$eval_data \
  "bash src/evaluate.sh gpt2 wordlength_eval_data"

# Evaluate on checkpoints of the continued-pretraining model.
# Change the step numbers below to reflect the checkpoints you saved.
for step in 2000 4000 6000 8000 10000; do
  model_save_dir=${dataset_name}_seed${seed}
  run_name=eval_${model_save_dir}_step${step}
  train_dir=train_${model_save_dir}
  model_dir=${model_save_dir}/checkpoint-${step}
  cl run \
    --name $run_name \
    --request-docker-image sangxie513/cs324-p2-codalab-gpu \
    --request-memory 32g \
    :src \
    :$eval_data \
    ${model_save_dir}:$train_dir/${model_save_dir} \
    "bash src/evaluate.sh ${model_dir} ${eval_data}"
done
