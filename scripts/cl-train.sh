#!/bin/bash

# Run `./train.sh` on CodaLab.
# Feel free to modify this script.

set -x

dataset_name=openwebtext_wordlength
seed=1111 # random seed
run_name=train_${dataset_name}_seed${seed}

# TODO: change these hyperparameters.
# Note: make sure that `gradient_accumulation_steps * max_steps <= 100000`
model_name_or_path=gpt2
per_device_train_batch_size=4
gradient_accumulation_steps=1
max_steps=10000
learning_rate=1e-6
warmup_steps=0
save_steps=2000

cl run \
  --name $run_name \
  --request-docker-image sangxie513/cs324-p2-codalab-gpu \
  --request-memory 80g \
  --request-gpus 1 \
  --request-cpus 8 \
  ${dataset_name}:preprocess_$dataset_name/${dataset_name} \
  :src \
  "bash src/train.sh $dataset_name $seed --model_name_or_path ${model_name_or_path} --per_device_train_batch_size ${per_device_train_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --max_steps ${max_steps} --learning_rate ${learning_rate} --warmup_steps ${warmup_steps} --save_steps ${save_steps} --fp16"
