import datetime
import pickle

from cs336_basics import train_bpe

if __name__ == "__main__":
    data_dir = "/home/liyang/github repos/cs336-2025/assignment1-basics/data"
    for data_set, vocab_size in [
        ("owt_valid", 32000),
        ("owt_train", 32000),
    ]:
        tic = datetime.datetime.now()
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
