import os
from typing import BinaryIO
import multiprocessing
import regex as re
import collections
import cProfile
import pickle
import time
import heapq
import dataclasses

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


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
        for chunk in re.split("|".join(special_tokens), text):
            for m in re.finditer(PAT, chunk):
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
    num_chunks = multiprocessing.cpu_count() - 1
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

    vocab = {i: tk.encode("utf-8") for i, tk in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pair_counter = collections.Counter()
    for tokens, count, _ in corpus:
        num_tokens = len(tokens)
        if num_tokens < 2:
            continue
        for i in range(num_tokens - 1):
            pair_counter[tuple(tokens[i : i + 2])] += count

    pair_freqs = PairFrequencies()
    for pair, count in pair_counter.items():
        pair_freqs.update(pair, count)

    tic = time.time()
    merges = []
    token_id = len(vocab)
    while token_id < vocab_size:
        top_pair = pair_freqs.pop()
        merge = b"".join(top_pair)
        vocab[token_id] = merge
        token_id += 1
        merges.append(top_pair)

        if len(vocab) % 5000 == 0:
            print(
                f"len(vocab) {len(vocab)} used {(time.time() - tic) / 60 / 60:.2f} hours"
            )
            tic = time.time()

        for chunk in corpus:
            tokens, count, num_tokens = chunk
            if num_tokens < 2:
                continue
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
                        pair_freqs.update((left, curr), -count)
                        pair_freqs.update((left, merge), count)
                    if j + 2 < num_tokens:
                        right = tokens[j + 2]
                        pair_freqs.update((next, right), -count)
                        pair_freqs.update((merge, right), count)
                    i += 1
                    j += 2
                    chunk[2] -= 1
            
            tokens[:] = tokens[:i]

    print(f"len(vocab) {len(vocab)} used {(time.time() - tic) / 60 / 60:.2f} hours")

    return vocab, merges


if __name__ == "__main__":
    data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
    run_profile = True
    if run_profile:
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
    else:
        for data_set, vocab_size in [
            # ("TinyStoriesV2-GPT4-valid", 10000),
            # ("TinyStoriesV2-GPT4-train", 10000),
            # ("owt_valid", 32000),
            ("owt_train", 32000),
        ]:
            tic = time.time()
            data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
            vocab, merges = train_bpe(
                f"{data_dir}/{data_set}.txt",
                vocab_size=vocab_size,
                special_tokens=["<|endoftext|>"],
            )
            print(f"total time elapsed {(time.time() - tic) / 60 / 60:.2f} hours")
            with open(f"{data_dir}/{data_set}-vocab.pickle", "wb") as f:
                pickle.dump(vocab, f)
            with open(f"{data_dir}/{data_set}-merges.pickle", "wb") as f:
                pickle.dump(merges, f)
