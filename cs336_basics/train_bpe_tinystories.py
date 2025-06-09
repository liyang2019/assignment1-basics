import train_bpe
import datetime
import pickle

if __name__ == "__main__":
    data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
    for data_set, vocab_size in [
        ("TinyStoriesV2-GPT4-valid", 10000),
        ("TinyStoriesV2-GPT4-train", 10000),
    ]:
        tic = datetime.datetime.now()
        data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
        vocab, merges = train_bpe.train_bpe(
            f"{data_dir}/{data_set}.txt",
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
        print(f"total time elapsed {datetime.datetime.now() - tic} hours")
        with open(f"{data_dir}/{data_set}-vocab.pickle", "wb") as f:
            pickle.dump(vocab, f)
        with open(f"{data_dir}/{data_set}-merges.pickle", "wb") as f:
            pickle.dump(merges, f)
