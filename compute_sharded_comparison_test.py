import os

import math
import random 

import numpy as np 
from scipy.stats import binom
from scipy.stats import t as tdist

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import GPUtil
from multiprocessing import Process, Queue

from tqdm import tqdm

import json

import fire
from utils.data_reading import read_jsonl_file

os.environ['TOKENIZERS_PARALLELISM'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flatten = lambda l : [x for s in l for x in s]
shuffle = lambda l : random.sample(l, k=len(l))

def load_dataset(dataset_path):
    
    # For loading a JSON-serialized list of examples.
    if dataset_path.endswith(".json"):
        print("loading from json...")
        with open(dataset_path, "r") as f:
            data = f.read()
            examples = json.loads(data)
            return examples

    # For loading a dataset where each example is on its own line.
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    return lines

def compute_logprob_of_token_sequence(tokens, model, context_len=2048, stride=1024, device=0):
  """
  Approximates logp(tokens) by sliding a window over the tokens with a stride.
  """
  inputs  = tokens[:-1]
  targets = tokens[1:]

  logp = torch.zeros((1, 1), dtype=torch.float32).to("cuda")

  # compute the smallest multiple k of s so that t <= ks + c.
  t = len(inputs); c = context_len; s = stride
  k = math.ceil(max(0, t - c) / s)
  all_logps = []
  for j in range(k + 1):
    start    = s * j
    end      = min(s * j + c, t)
    rel_offs = max(0, c - s) if j > 0 else 0

    w_inp = inputs[start:end]; w_inp = torch.tensor(w_inp).to("cuda")
    w_trg = targets[start:end]; w_trg = torch.tensor(w_trg).to("cuda")

    model.eval()
    with torch.no_grad():
      out = model(torch.unsqueeze(w_inp, 0))
      logps = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
      logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
      logp += logps[rel_offs:].sum()

    del w_inp
    del w_trg
    torch.cuda.empty_cache()

  return logp.item()

def worker(model_name_or_path,
           context_len,
           stride, tokens, shard_id, is_canonical, m):
    
    # Load model.
    # main_queue.put(True)
    # Wait for inference requests.

        # Compute logprob of tokens.
    logprob = compute_logprob_of_token_sequence(tokens,
                                                    m, 
                                                    context_len, 
                                                    stride,
                                        )

        # Send result to main process.
    return logprob, shard_id, is_canonical
        

def main(model_name_or_path="/home/liuhui/llms/l2/Llama-2-7b-chat-hf",
         dataset_path="/home/liuhui/unify/data/LIAR-PLUS/test.jsonl" ,
         context_len=2048,
         stride=512,
         num_shards=6,
         permutations_per_shard=10,
         random_seed=0,
         log_file_path="result.log",
         max_examples=200):

    # Set random seed(s).
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the dataset.
    example = read_jsonl_file(os.path.join(dataset_path))
    examples = [ex["statement"] for ex in example]
    examples = examples[:max_examples]
    num_examples = len(examples)
    print(f"Loaded {num_examples} examples from {dataset_path}")

    # Load tokenizer and tokenize the examples.
    t = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_examples = [t.encode(ex) for ex in examples]
    print(tokenized_examples)
    m = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    # m.cuda()
    # Launch a Process for each GPU.
    # gpus = 1
    canonical_logprobs = [None for _ in range(num_shards)]
    shuffled_logprobs  = [[] for _ in range(num_shards)]
    # for i in range(num_workers):
    #     p = Process(target=worker, args=(model_name_or_path,
    #                                      context_len,
    #                                      stride,
    #                                      123234,
    #                                      main_queue,
    #                                      worker_queues[i]))
    #     processes.append(p)
    #     p.start()
    #
    # Wait until each GPU has loaded a model.
    num_ready = 0
    # while num_ready < num_workers:
    #     gpu_id, is_ready = main_queue.get()
    #     print(f"GPU {gpu_id} loaded model.")
    #     num_ready += 1
    #
    # Issue requests to all worker queues, round-robin style.
    
    # Compute the number of examples for each shard.
    shard_counts = [(x + 1 if i < num_examples % num_shards else x) 
       for i, x in enumerate([num_examples // num_shards] * num_shards)]
    shard_counts = np.asarray(shard_counts)

    # Compute the starting index (into the list of examples) for each shard.
    shard_example_indices = [0] + np.cumsum(shard_counts).tolist()
    for i, (start, end) in enumerate(zip(shard_example_indices, shard_example_indices[1:])):
        print("{} shard \n".format(i))
        shard = tokenized_examples[start:end]
        # Logprobs in canonical order.
        logprob, shard_id, is_canonical = worker(model_name_or_path, context_len, stride,
        flatten(shard), i,  True, m)
        canonical_logprobs[shard_id] = logprob
        # Logprobs in shuffled order(s). 
        for j in range(permutations_per_shard):
            print("{} permutation \n".format(j))
            logprob, shard_id, is_canonical = worker(model_name_or_path, context_len, stride,
                   flatten(shuffle(shard)), i, False, m)
            print(logprob)
            shuffled_logprobs[shard_id].append(logprob)
    # Wait on requests.
    # total_work = num_shards * (1 + permutations_per_shard)
    # pbar = tqdm(total=total_work)

    # Calculate p-value.
    canonical_logprobs = np.asarray(canonical_logprobs)
    shuffled_logprobs  = np.asarray(shuffled_logprobs)
    
    # T-test.
    diffs = canonical_logprobs - shuffled_logprobs.mean(axis=1)
    print( diffs )
    z = np.mean(diffs) / np.std(diffs) * np.sqrt(len(diffs))
    pval = 1 - tdist.cdf(z, df=len(diffs)-1)
    print(f"{pval=}")

    # Log.
    if log_file_path is not None:
        print(f"Writing logprobs to: {log_file_path}")
        with open(f"{log_file_path}", 'w') as f:
            f.write(json.dumps({
                'pval': pval, 
                'permutations_per_shard': permutations_per_shard,
                'num_shards': num_shards,
                'canonical_logprobs': canonical_logprobs.tolist(),
                'shuffled_logprobs': shuffled_logprobs.tolist(),
            }))

if __name__ == '__main__':
  fire.Fire(main)
#
# python compute_sharded_stat.py /home/liuhui/llms/l2/Llama-2-7b-chat-hf /home/liuhui/unify/data/Constraint/test.jsonl \
# --context_len 1024 \
# --stride 512 \
# --num_shards 50 \
# --permutations_per_shard 100 \
# --log_file_path "result.log"