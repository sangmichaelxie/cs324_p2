#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "Main entry point for preprocessing data."
  echo "Usage:"
  echo
  echo "    $0 <input_dataset_name (e.g., openwebtext)> <output_dataset_name (e.g., openwebtext_wordlength)> <tokenizer_name (e.g., gpt2)> <num_total_chunks (e.g., 64)> <num_chunks (e.g., 10)>"
  echo
  exit 1
fi

CACHE=./cache
mkdir -p $CACHE

input_dataset_name=$1
output_dataset_name=$2
tokenizer_name=$3
num_total_chunks=${4:-64}
num_chunks=${5:-10} # only process a subset of the total chunks. Set to num_total_chunks to process full dataset

mkdir -p $output_dataset_name

# Process the chunks in parallel
for ((CHUNK_IDX=0; CHUNK_IDX < $num_chunks; CHUNK_IDX++)); do
  python src/process_data.py \
    --data_dir $input_dataset_name \
    --output_dir $output_dataset_name \
    --total_chunks $num_total_chunks \
    --chunk_idx $CHUNK_IDX &
done

wait

# Tokenize the data in chunks without merging
python src/tokenize_data.py \
  --data_dir $output_dataset_name \
  --cache_dir ${CACHE} \
  --model_name ${tokenizer_name} \
