import datetime
import pickle
import random
import time
import numpy as np

from cs336_basics import train_bpe
from cs336_basics import tokenizer
from cs336_basics import common

DATA_DIR = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"


def train_tokenizer():
    for data_set, vocab_size in [
        ("owt_valid", 32000),
        ("owt_train", 32000),
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
        f"{DATA_DIR}/owt_train-vocab.pickle",
        f"{DATA_DIR}/owt_train-merges.pickle",
        ["<|endoftext|>"],
    )
    for data_set in [
        "owt_valid",
        # "owt_train",
    ]:
        tic = datetime.datetime.now()
        with open(f"{DATA_DIR}/{data_set}.txt", "r") as f:
            doc = f.read()
        token_ids = np.array(bpe_tokenizer.encode(doc), dtype=np.uint16)
        print(f"total time elapsed for {data_set} = {datetime.datetime.now() - tic}")
        np.save(f"{DATA_DIR}/{data_set}-token_ids.npy", token_ids)


def encode_dataset():
    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/owt_train-vocab.pickle",
        f"{DATA_DIR}/owt_train-merges.pickle",
        ["<|endoftext|>"],
    )

    special_token = "<|endoftext|>".encode("utf-8")
    common.encode_dataset(
        bpe_tokenizer,
        special_token,
        [
            "owt_valid",
            "owt_train",
        ],
    )


def estimate_compression_ratio():
    with open(f"{DATA_DIR}/owt_valid.txt") as f:
        docs = f.read().split("<|endoftext|>")
        samples = random.sample(docs, 10)

    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/owt_train-vocab.pickle",
        f"{DATA_DIR}/owt_train-merges.pickle",
        ["<|endoftext|>"],
    )

    token_ids = []
    for sample in samples:
        token_ids.append(bpe_tokenizer.encode(sample))

    num_bytes = sum(len(s.encode("utf-8")) for s in samples)
    num_tokens = sum(len(tks) for tks in token_ids)
    print(f"compression ratio = {num_bytes / num_tokens}")

    ts_bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-vocab.pickle",
        f"{DATA_DIR}/TinyStoriesV2-GPT4-train-merges.pickle",
        ["<|endoftext|>"],
    )

    token_ids = []
    for sample in samples:
        token_ids.append(ts_bpe_tokenizer.encode(sample))

    num_bytes = sum(len(s.encode("utf-8")) for s in samples)
    num_tokens = sum(len(tks) for tks in token_ids)
    print(f"compression ratio using TinyStories tokenizer = {num_bytes / num_tokens}")


def estimate_throughtput():
    with open(f"{DATA_DIR}/owt_valid.txt") as f:
        doc = f.read()

    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{DATA_DIR}/owt_train-vocab.pickle",
        f"{DATA_DIR}/owt_train-merges.pickle",
        ["<|endoftext|>"],
    )

    tic = time.time()
    token_ids = bpe_tokenizer.encode(doc)
    delta = time.time() - tic

    num_bytes = len(doc.encode("utf-8"))
    print(f"throughtpu estimated on {num_bytes} bytes = {num_bytes / delta}")


if __name__ == "__main__":
    # train_tokenizer()
    # estimate_compression_ratio()
    # estimate_throughtput()
    encode_dataset()
