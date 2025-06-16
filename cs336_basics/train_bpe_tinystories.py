import datetime
import pickle
import random
import numpy as np
import os

from cs336_basics import train_bpe
from cs336_basics import tokenizer
from cs336_basics import common

DATA_DIR = common.DATA_DIR


def train_tokenizer():
    for data_set, vocab_size in [
        ("TinyStoriesV2-GPT4-valid", 10000),
        ("TinyStoriesV2-GPT4-train", 10000),
    ]:
        tic = datetime.datetime.now()
        vocab, merges = train_bpe.train_bpe(
            f"{DATA_DIR}/{data_set}.txt",
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
        print(f"total time elapsed for {data_set} = {datetime.datetime.now() - tic}")
        with open(f"{DATA_DIR}/{data_set}-vocab.pickle", "wb") as f:
            pickle.dump(vocab, f)
        with open(f"{DATA_DIR}/{data_set}-merges.pickle", "wb") as f:
            pickle.dump(merges, f)


def encode_dataset():
    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-vocab.pickle",
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-merges.pickle",
        ["<|endoftext|>"],
    )

    special_token = "<|endoftext|>".encode("utf-8")
    common.encode_dataset(
        bpe_tokenizer,
        special_token,
        [
            "TinyStoriesV2-GPT4-valid",
            # "TinyStoriesV2-GPT4-train",
        ],
    )


def estimate_compression_ratio():
    with open(f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt") as f:
        docs = f.read().split("<|endoftext|>")
        samples = random.sample(docs, 10)

    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-vocab.pickle",
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-merges.pickle",
        ["<|endoftext|>"],
    )

    token_ids = []
    for sample in samples:
        token_ids.append(bpe_tokenizer.encode(sample))

    num_bytes = sum(len(s.encode("utf-8")) for s in samples)
    num_tokens = sum(len(tks) for tks in token_ids)
    print(f"compression ratio = {num_bytes / num_tokens}")


if __name__ == "__main__":
    # train_tokenizer()
    # estimate_compression_ratio()
    encode_dataset()
