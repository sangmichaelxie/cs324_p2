from datasets import load_dataset, load_from_disk

import numpy as np
import re
import os
from pathlib import Path
import json
import shutil
import multiprocessing

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def get_paths(split='train'):
    paths = list(afs_dir.glob(f'{split}.jsonl_*'))
    paths = [str(p) for p in paths]
    # sort by numbers
    path_idxs = [int(p.split('.jsonl_')[1]) for p in paths]
    sort_idxs = np.argsort(path_idxs)
    paths = list(np.asarray(paths)[sort_idxs])

    return paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tokenize the processed data')
    parser.add_argument('--data_dir', type=str, help="where the data is saved")
    parser.add_argument('--cache_dir', type=str, help="cache for intermediate files")
    parser.add_argument('--model_name', type=str, help="Huggingface model name (e.g., gpt2-medium)")
    args = parser.parse_args()

    model_name_or_path = args.model_name
    afs_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    use_fast_tokenizer = True
    model_revision = 'main'
    use_auth_token = False

    json_path_train_ls = get_paths(split='train')
    json_path_val_ls = get_paths(split='val')

    tokenizer_kwargs = {
        "cache_dir": str(cache_dir),
        "use_fast": use_fast_tokenizer,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    block_size_10MB = 10<<20
    data_files = {
        'train': json_path_train_ls,
        'validation': json_path_val_ls,
            }
    json_cache = cache_dir / 'json_cache'
    json_cache.mkdir(exist_ok=True)
    datasets = load_dataset('json', data_files=data_files, chunksize=block_size_10MB, cache_dir=str(json_cache))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    curr_save_dir = afs_dir / 'tokenized'
    if not curr_save_dir.exists():
        curr_save_dir.mkdir(exist_ok=True, parents=True)
        datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=True,
        )
        datasets.save_to_disk(str(curr_save_dir))
    else:
        datasets = load_from_disk(str(curr_save_dir))

    block_size = 1024
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
    )
    curr_save_dir = afs_dir / 'tokenized_grouped'
    curr_save_dir.mkdir(exist_ok=True)
    datasets.save_to_disk(str(curr_save_dir))

    # remove tokenize dir to save space
    tokenized_dir = afs_dir / 'tokenized'
    shutil.rmtree(str(tokenized_dir))

