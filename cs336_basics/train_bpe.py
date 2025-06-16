import os
from typing import BinaryIO
import multiprocessing
import regex as re
import collections
import cProfile
import heapq
import dataclasses
import datetime

from cs336_basics import common


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pre_tokenize(start, end, input_path, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
        freqs = collections.Counter()
        special_tokens_escaped = sorted(
            [re.escape(st) for st in special_tokens], key=lambda p: -len(p)
        )
        chunks = (
            re.split(f"{'|'.join(special_tokens_escaped)}", text)
            if special_tokens_escaped
            else [text]
        )
        for chunk in chunks:
            for m in re.finditer(common.PAT, chunk):
                key = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
                freqs[key] += 1
        return freqs


@dataclasses.dataclass
class Entry:
    pair: tuple[bytes, bytes]
    count: int
    removed: bool = False

    def __lt__(self, that):
        if self.count > that.count:
            return True
        if self.count == that.count:
            return self.pair > that.pair
        return False


class PairFrequencies:

    def __init__(self):
        self.pq = []
        self.entry_finder = {}

    def update(self, pair, count):
        if pair not in self.entry_finder:
            new_entry = Entry(pair, count)
        else:
            old_entry = self.remove(pair)
            new_entry = Entry(pair, old_entry.count + count)
        heapq.heappush(self.pq, new_entry)
        self.entry_finder[pair] = new_entry

    def remove(self, pair):
        entry = self.entry_finder.pop(pair)
        entry.removed = True
        return entry

    def pop(self):
        while self.pq:
            entry = heapq.heappop(self.pq)
            if not entry.removed:
                del self.entry_finder[entry.pair]
                return entry.pair
        raise ValueError("Empty pq")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_chunks = multiprocessing.cpu_count() - 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8")
        )

    with multiprocessing.Pool(num_chunks) as p:
        args = zip(
            boundaries[:-1],
            boundaries[1:],
            [input_path] * num_chunks,
            [special_tokens] * num_chunks,
        )
        corpus = sum(p.starmap(_pre_tokenize, args), start=collections.Counter())
        corpus = [[list(k), v, len(k)] for k, v in corpus.items()]

    print(f"corpus size: {len(corpus)}")

    vocab = {i: tk.encode("utf-8") for i, tk in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pair_counter = collections.Counter()
    pair_locations = collections.defaultdict(set)
    for cid, (tokens, count, num_tokens) in enumerate(corpus):
        if num_tokens < 2:
            continue
        for i in range(num_tokens - 1):
            pair = tuple(tokens[i : i + 2])
            pair_counter[pair] += count
            pair_locations[pair].add(cid)

    pair_freqs = PairFrequencies()
    for pair, count in pair_counter.items():
        pair_freqs.update(pair, count)

    tic = datetime.datetime.now()
    merges = []
    token_id = len(vocab)
    while token_id < vocab_size:
        top_pair = pair_freqs.pop()
        merge = b"".join(top_pair)
        vocab[token_id] = merge
        token_id += 1
        merges.append(top_pair)

        updates = collections.defaultdict(int)
        for cid in pair_locations.pop(top_pair):
            pre_token = corpus[cid]
            tokens, count, num_tokens = pre_token
            i, j = 0, 0
            while j < num_tokens:
                if j == num_tokens - 1 or (tokens[j], tokens[j + 1]) != top_pair:
                    tokens[i] = tokens[j]
                    i += 1
                    j += 1
                else:
                    curr, next = tokens[j], tokens[j + 1]
                    tokens[i] = merge
                    if i > 0:
                        left = tokens[i - 1]
                        updates[(left, curr)] -= count
                        updates[(left, merge)] += count
                        pair_locations[(left, merge)].add(cid)
                    if j + 2 < num_tokens:
                        right = tokens[j + 2]
                        updates[(next, right)] -= count
                        updates[(merge, right)] += count
                        pair_locations[(merge, right)].add(cid)
                    i += 1
                    j += 2
                    pre_token[2] -= 1
            tokens[:] = tokens[:i]

        for pair, count in updates.items():
            pair_freqs.update(pair, count)

        if token_id % 5000 == 0 or token_id == vocab_size:
            print(f"len(vocab) {len(vocab)} used {datetime.datetime.now() - tic}")
            tic = datetime.datetime.now()
            print(f"num updates {len(updates)}")

    return vocab, merges


if __name__ == "__main__":
    data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
    cProfile.run(
        f"""
vocab, merges = train_bpe(
"{data_dir}/TinyStoriesV2-GPT4-valid.txt",
vocab_size=10000,
special_tokens=["<|endoftext|>"],
)
            """,
        sort="tottime",
    )
