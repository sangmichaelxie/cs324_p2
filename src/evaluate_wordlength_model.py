import numpy as np
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
)
from tqdm import tqdm


CONFIG_NAME = "gpt2"

config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }

tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the trained wordlength model')
    parser.add_argument('--model_dir', type=str, help="where the trained model is saved. Can also input a huggingface model name like gpt2 to get the default pretrained model.")
    parser.add_argument('--eval_data_dir', type=str, help="directory of the evaluation data")
    parser.add_argument('--eval_num_examples', type=int, default=None, help="number of examples to evaluate on")
    parser.add_argument('--cuda', action='store_true', help="number of examples to evaluate on")
    args = parser.parse_args()

    # load the model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, **tokenizer_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                from_tf=False,
                cache_dir=None,
                revision="main",
                use_auth_token=None,
            )
    if args.cuda:
        model.cuda()

    eval_data_dir = Path(args.eval_data_dir)

    metrics = []
    # calculate metrics over all seeds of the evaluation dataset
    for eval_data_path in eval_data_dir.iterdir():
        generations = []
        labels = []
        predicted_labels = []
        with open(eval_data_path, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                if args.eval_num_examples and i >= args.eval_num_examples:
                    break
                row = json.loads(line)
                prefix = row['text']
                num_words = int(row['num_words'])
                labels.append(num_words)

                # tokenize
                if args.cuda:
                    input_ids = tokenizer.encode(prefix, return_tensors="pt").cuda()
                else:
                    input_ids = tokenizer.encode(prefix, return_tensors="pt")

                sample_output = model.generate(input_ids,
                                   do_sample=True,
                                   max_length=200,
                                   top_k=5)
                # un-tokenize
                decoded_output = tokenizer.decode(sample_output[0], skip_special_tokens=False)

                # parse the output
                decoded_output = decoded_output.split('<text>')[1]
                decoded_output = decoded_output.split('<len>')[0].strip()
                generations.append(decoded_output)

                # count the number of words
                generated_num_words = len(word_tokenize(decoded_output))
                predicted_labels.append(generated_num_words)

        # compute the metric
        labels = np.asarray(labels)
        predicted_labels = np.asarray(predicted_labels)
        generations = np.asarray(generations)
        print(f"Model: {args.model_dir}")
        print("========= Some generations ======")
        idxs = np.arange(len(generations))
        np.random.shuffle(idxs)

        # print some generations
        for num_words, generated_num_words, generation in zip(labels[idxs[:10]], predicted_labels[idxs[:10]], generations[idxs[:10]]):
            print(f"TARGET NUM_WORDS: {num_words}, GENERATED_NUM_WORDS: {generated_num_words}, GENERATION: {generation}")

        print("=================================")
        metric = np.mean(np.abs(labels - predicted_labels))
        print(f"Metric for wordlength: {metric:.4f}")
        metrics.append(metric)
    print("============================")
    print("============================")

    print(f"Mean: {np.mean(metrics)}")
    print(f"Std: {np.std(metrics)}")

    results = {i: metrics[i] for i in range(len(metrics))}
    with open('results.json', 'w') as f:
        f.write(json.dumps(results))
