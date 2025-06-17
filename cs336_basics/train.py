import transformer
import wandb
import dataclasses
import click
import torch
import numpy as np

from cs336_basics import common

@click.command()
@click.option("--vocab-size", type=int, default=10000)
@click.option("--d-model", type=int, default=512)
@click.option("--d-ff", type=int, default=1344)
@click.option("--num-heads", type=int, default=16)
@click.option("--num-layers", type=int, default=4)
@click.option("--context-length", type=int, default=256)
@click.option("--rope-theta", type=int, default=10000)
def run_train(
    vocab_size: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_layers: int,
    context_length: int,
    rope_theta: float,
):
    model = transformer.TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        device="cuda",
        dtype=torch.bfloat16,
    )
    data = np.load(f'{common.DATA_DIR}/')
