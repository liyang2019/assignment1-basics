import os
from typing import BinaryIO
import multiprocessing
import regex as re
import collections
import cProfile
import pickle
import time

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

    vocab = {i: tk.encode("utf-8") for i, tk in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pair_freqs = collections.Counter()
    for tokens, count in corpus.items():
        if len(tokens) < 2:
            continue
        for i in range(len(tokens) - 1):
            pair_freqs[tokens[i : i + 2]] += count

    merges = []
    while len(vocab) < vocab_size:
        top_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        del pair_freqs[top_pair]
        merge = b"".join(top_pair)
        vocab[len(vocab)] = merge
        merges.append(top_pair)

        merged_corpus = {}
        for tokens, count in corpus.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i : i + 2] == top_pair:
                    if new_tokens:
                        pair_freqs[(new_tokens[-1], tokens[i])] -= count
                        pair_freqs[(new_tokens[-1], merge)] += count
                    if i + 2 < len(tokens):
                        pair_freqs[(tokens[i + 1], tokens[i + 2])] -= count
                        pair_freqs[(merge, tokens[i + 2])] += count
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            merged_corpus[tuple(new_tokens)] = count
        corpus = merged_corpus

    return vocab, merges


if __name__ == "__main__":
    #     cProfile.run(
    #         """
    # vocab, merges = train_bpe(
    #     "/Users/liyang2029/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    # )
    #         """,
    #         sort="tottime",
    #     )
    tic = time.time()
    data_dir = "/Users/liyang2029/assignment1-basics/data/"
    data_set = "owt_valid"
    vocab, merges = train_bpe(
        f"{data_dir}/{data_set}.txt", vocab_size=32000, special_tokens=["<|endoftext|>"]
    )
    print('time elapsed', time.time() - tic)
    with open(f"{data_dir}/{data_set}-vocab.pickle", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{data_dir}/{data_set}-merges.pickle", "wb") as f:
        pickle.dump(merges, f)
