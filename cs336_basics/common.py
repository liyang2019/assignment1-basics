from typing import Iterator
import regex as re
import datetime
import os
import numpy as np

import tokenizer

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

DATA_DIR = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"


def encode_dataset(
    bpe_tokenizer: tokenizer.Tokenizer,
    special_token: bytes,
    data_sets: list[str],
    chunk_size: int = 128 * 2**20,  # 128MB
    mini_chunk_size: int = 4096,  # 4K
):
    chunk_size = 128 * 2**20  # 128MB per read
    mini_chunk_size = 4096  # 4K per read ahead for special token
    for data_set in data_sets:
        all_token_ids = []
        start = 0
        tic = datetime.datetime.now()
        with open(f"{DATA_DIR}/{data_set}.txt", "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            while start < file_size:
                end = min(start + chunk_size, file_size)
                f.seek(end)
                while True:
                    ahead = f.read(mini_chunk_size)
                    if ahead == b"":
                        break
                    found_at = ahead.find(special_token)
                    if found_at != -1:
                        end += found_at
                        break
                    end += mini_chunk_size
                f.seek(start)
                chunk = f.read(end - start)
                token_ids = np.array(
                    bpe_tokenizer.encode(chunk.decode("utf-8")), dtype=np.uint16
                )
                all_token_ids.append(token_ids)
                print(
                    f"{data_set} time elapsed for {end - start} bytes "
                    f"= {datetime.datetime.now() - tic}"
                )
                tic = datetime.datetime.now()
                start = end

        all_token_ids = np.concatenate(all_token_ids)
        np.save(f"{DATA_DIR}/{data_set}-token_ids_chunk.npy", all_token_ids)
