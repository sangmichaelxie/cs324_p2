'''
Process the openwebtext dataset in chunks. Can parallelize the execution of
this script across chunks and then call this script with the --reduce_step flag
to merge the chunks (if needed). Typically, HuggingFace will accept the data in
chunks (as a list of data paths).

Call this script with an incomplete number of chunk indices to process only
part of the dataset.

The script currently tags each sentence with its length.
'''
from pathlib import Path
from tqdm import tqdm
import subprocess
import json
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize


def linecount(filename):
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            count += 1
    return count


def merge_files(filepaths, outpath):
    with open(outpath, 'w') as outfile:
        for fname in tqdm(filepaths):
            with open(fname, 'r') as infile:
                for line in infile:
                    outfile.write(line)


def write_record(record, fp):
    record_str = json.dumps(record)
    fp.write(f"{record_str}\n")
    fp.flush()


def tag_word_length(doc):
    # split the document into sentences
    sents = sent_tokenize(json.loads(doc)['text'])
    new_sents = []
    for sent in sents:
        num_words = len(word_tokenize(sent))
        new_sent = f"<len> {num_words} <text> {sent}"
        new_sents.append(new_sent)
    return " ".join(new_sents)


def process_fn(save_dir, split, txt_data):
    print("Processing data")

    # count lines
    count = linecount(txt_data)

    chunk_size = count // args.total_chunks
    chunk_start_idx = chunk_size * args.chunk_idx
    chunk_end_idx = chunk_size * (args.chunk_idx + 1)
    chunk_path = f"{path}_{args.chunk_idx}"
    with open(chunk_path, 'w') as fw:
        with open(txt_data, 'r') as fr:
            with tqdm(total=chunk_size) as pbar:
                for i, doc in enumerate(fr):
                    if i < chunk_start_idx:
                        continue
                    elif i >= chunk_end_idx:
                        break
                    else:
                        ############################
                        # TODO Replace with your own function for a custom dataset
                        doc = tag_word_length(doc)
                        write_record({'text': doc}, fw)
                        ############################

                        # move the progress bar 1 unit
                        pbar.update(1)
    return chunk_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='preprocess data in chunks')
    parser.add_argument('--data_dir', type=str, help="where the openwebtext data is downloaded")
    parser.add_argument('--output_dir', type=str, help="where to output the processed files")
    parser.add_argument('--chunk_idx', type=int, default=0, help="which chunk to generate")
    parser.add_argument('--total_chunks', type=int, default=8, help="total number of chunks")
    parser.add_argument('--reduce_step', action='store_true', help="whether to do the reduce step instead of the mapping step")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    splits = ['train', 'val']

    if not args.reduce_step:
        for split in splits:
            path = save_dir / f'{split}.jsonl'
            txt_path = data_dir / f'{split}.jsonl'
            save_path = process_fn(save_dir, split, txt_path)
            save_path = Path(save_path)
    else:
        # merge data (optional, not necessarily recommended)
        for split in splits:
            path = save_dir / f'{split}.jsonl'
            filepaths = [f"{path}_{i}" for i in range(args.total_chunks)]
            merge_files(filepaths, path)

