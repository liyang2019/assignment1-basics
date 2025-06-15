import os
from typing import Iterable, Iterator
import multiprocessing
import regex as re
import collections
import cProfile
import pickle
import time
import heapq
import dataclasses
import functools
import datetime

from cs336_basics import common


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.id_by_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_escaped = sorted(
            [re.escape(st) for st in self.special_tokens], key=lambda p: -len(p)
        )
        self.special_tokens_pattern = re.compile(
            f"({'|'.join(self.special_tokens_escaped)})"
        )
        self.mini_chunk_size = 4096  # Read ahead by 4k chars at a time

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _merge(self, tokens: list[bytes]) -> list[bytes]:
        num_tokens = len(tokens)
        seen_pairs = set()
        for i in range(len(tokens) - 1):
            seen_pairs.add((tokens[i], tokens[i + 1]))
        for pair in self.merges:
            if num_tokens < 2:
                return
            if pair not in seen_pairs:
                continue
            merge = b"".join(pair)
            i, j = 0, 0
            while j < num_tokens:
                if j == num_tokens - 1 or (tokens[j], tokens[j + 1]) != pair:
                    tokens[i] = tokens[j]
                    j += 1
                else:
                    tokens[i] = merge
                    if i > 0:
                        seen_pairs.add((tokens[i - 1], merge))
                    if j + 2 < num_tokens:
                        seen_pairs.add((merge, tokens[j + 2]))
                    j += 2
                i += 1
            num_tokens = i
            tokens[:] = tokens[:num_tokens]

    def _encode(self, tokens: list[bytes]) -> Iterator[int]:
        self._merge(tokens)
        for token in tokens:
            yield self.id_by_token[token]

    def encode(self, text: str) -> list[int]:
        token_ids = []
        chunks = (
            re.split(self.special_tokens_pattern, text)
            if self.special_tokens
            else [text]
        )
        for chunk in chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.id_by_token[chunk.encode("utf-8")])
                continue
            for m in re.finditer(common.PAT, chunk):
                tokens = [bytes([b]) for b in m.group(0).encode("utf-8")]
                token_ids.extend(self._encode(tokens))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            chunks = (
                re.splititer(self.special_tokens_pattern, text)
                if self.special_tokens
                else [text]
            )
            for chunk in chunks:
                if chunk in self.special_tokens:
                    yield self.id_by_token[chunk.encode("utf-8")]
                    continue
                for m in re.finditer(common.PAT, chunk):
                    tokens = [bytes([b]) for b in m.group(0).encode("utf-8")]
                    yield from self._encode(tokens)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab.get(x, "�".encode("utf-8")) for x in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")
