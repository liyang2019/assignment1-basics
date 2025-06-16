from typing import Iterable, Iterator
import regex as re
import collections
import pickle
import itertools

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

    @classmethod
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

    def encode(self, text: str) -> list[int]:
        locs_by_pre_token = collections.defaultdict(list)
        chunks = (
            re.split(self.special_tokens_pattern, text)
            if self.special_tokens
            else [text]
        )
        num_pre_tokens = 0
        for chunk in chunks:
            if chunk in self.special_tokens:
                locs_by_pre_token[chunk].append(num_pre_tokens)
                num_pre_tokens += 1
                continue
            for m in re.finditer(common.PAT, chunk):
                locs_by_pre_token[m.group(0)].append(num_pre_tokens)
                num_pre_tokens += 1
        corpus = []
        for pre_token, locs in locs_by_pre_token.items():
            if pre_token in self.special_tokens:
                tokens = [pre_token.encode("utf-8")]
            else:
                tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
            corpus.append([tokens, locs, len(tokens)])

        pair_locations = collections.defaultdict(set)
        for cid, (tokens, locs, num_tokens) in enumerate(corpus):
            if num_tokens < 2:
                continue
            for i in range(num_tokens - 1):
                pair = tuple(tokens[i : i + 2])
                pair_locations[pair].add(cid)

        for pair in self.merges:
            if pair not in pair_locations:
                continue
            merge = b"".join(pair)
            for cid in pair_locations.pop(pair):
                pre_token = corpus[cid]
                tokens, locs, num_tokens = pre_token
                i, j = 0, 0
                while j < num_tokens:
                    if j == num_tokens - 1 or (tokens[j], tokens[j + 1]) != pair:
                        tokens[i] = tokens[j]
                        i += 1
                        j += 1
                    else:
                        tokens[i] = merge
                        if i > 0:
                            left = tokens[i - 1]
                            pair_locations[(left, merge)].add(cid)
                        if j + 2 < num_tokens:
                            right = tokens[j + 2]
                            pair_locations[(merge, right)].add(cid)
                        i += 1
                        j += 2
                        pre_token[2] -= 1
                tokens[:] = tokens[:i]

        all_token_ids = [None] * num_pre_tokens
        for tokens, locs, _ in corpus:
            token_ids = [self.id_by_token[t] for t in tokens]
            for loc in locs:
                all_token_ids[loc] = token_ids
        return list(itertools.chain.from_iterable(all_token_ids))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab.get(x, "�".encode("utf-8")) for x in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")
