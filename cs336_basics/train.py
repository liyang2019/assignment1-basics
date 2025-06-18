import transformer
import wandb
import dataclasses
import click
import torch
import numpy as np
import datetime
import os

from cs336_basics import tokenizer
from cs336_basics import common


@click.command()
@click.option("--vocab-size", type=int, default=10000)
@click.option("--d-model", type=int, default=512)
@click.option("--d-ff", type=int, default=1344)
@click.option("--num-heads", type=int, default=16)
@click.option("--num-layers", type=int, default=4)
@click.option("--context-length", type=int, default=256)
@click.option("--rope-theta", type=int, default=10000)
@click.option("--num-iters", type=int, default=10000)
@click.option("--batch-size", type=int, default=128)
@click.option("--lr-schedule", type=str, default="cosine")
@click.option("--max-learning-rate", type=float, default=1e-3)
@click.option("--min-learning-rate", type=float, default=1e-4)
@click.option("--warmup-steps", type=int, default=1000)
@click.option("--warmdown-steps", type=int, default=8000)
@click.option("--cap-steps", type=int, default=9000)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--beta1", type=float, default=0.9)
@click.option("--beta2", type=float, default=0.95)
@click.option("--eps", type=float, default=1e-8)
@click.option("--grad-l2-max", type=float, default=7.0)
@click.option("--exp-name", type=str, default="default")
def run_train(
    vocab_size: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_layers: int,
    context_length: int,
    rope_theta: float,
    num_iters: int,
    batch_size: int,
    lr_schedule: str,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    warmdown_steps: int,
    cap_steps: int,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    grad_l2_max: float,
    exp_name: str,
):
    run = wandb.init(
        entity="liyang2029-meta",
        project="cs336-2025",
        name=exp_name,
        config=locals(),
    )
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = "cuda"
    model = transformer.TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        device=device,
        dtype=torch.bfloat16,
    )
    model.compile()
    optimizer = transformer.AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
    )
    data = np.load(
        f"{common.DATA_DIR}/TinyStoriesV2-GPT4-train-token_ids.npy",
        mmap_mode="r",
    )
    bpe_tokenizer = tokenizer.Tokenizer.from_files(
        f"{common.DATA_DIR}/TinyStoriesV2-GPT4-train-vocab.pickle",
        f"{common.DATA_DIR}/TinyStoriesV2-GPT4-train-merges.pickle",
        ["<|endoftext|>"],
    )
    checkpoint_dir = (
        f"./checkpoints/{datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"checkpoint_dir {checkpoint_dir!r}")
    model.train()
    tic = datetime.datetime.now()
    for iter in range(1, num_iters + 1):
        inputs, targets = transformer.get_batch(
            data, batch_size, context_length, device
        )
        outputs = model(inputs)
        loss = transformer.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        transformer.gradient_clipping(model.parameters(), l2_max=grad_l2_max)
        if lr_schedule == "cosine":
            lr = transformer.cosine_learning_rate_schedule(
                iter,
                max_learning_rate,
                min_learning_rate,
                warmup_steps,
                cap_steps,
            )
        elif lr_schedule == "wsd":
            lr = transformer.wsd_learning_rate_schedule(
                iter,
                max_learning_rate,
                min_learning_rate,
                warmup_steps,
                warmdown_steps,
                cap_steps,
            )
        optimizer.set_learning_rate(lr)
        optimizer.step()
        if iter % 10 == 0:
            toc = datetime.datetime.now()
            dt = toc - tic
            num_tokens = np.prod(inputs.shape)
            logs = {
                "lr": lr,
                "train/loss": float(loss),
                "train/throughout": num_tokens / dt.total_seconds(),
            }
            run.log(logs, step=iter)
            logs.update(
                {
                    "time": toc.strftime("%d-%m-%Y-%H:%M:%S"),
                    "iter": iter,
                }
            )
            print(
                "  ".join(
                    (f"{k} {v:0<10.6f}" if isinstance(v, float) else f"{k} {v:<10}")
                    for k, v in logs.items()
                )
            )
            tic = datetime.datetime.now()

        if iter % 100 == 0:
            print("test decoding =================")
            inputs = "Once upon a time there was"
            print(inputs)
            inputs = bpe_tokenizer.encode(inputs)
            model.eval()
            outputs = model.decode(
                torch.tensor(inputs, device=device),
                max_num_tokens=256,
                temperature=0,
            )
            outputs = bpe_tokenizer.decode(outputs.cpu().numpy().tolist())
            print(outputs)
            model.train()

    transformer.save_checkpoint(model, optimizer, iter, f"{checkpoint_dir}/iter-{iter}")
    run.finish()


if __name__ == "__main__":
    run_train()
