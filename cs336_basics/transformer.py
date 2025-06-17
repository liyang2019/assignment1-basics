import torch
from torch import nn
from einops import einsum, rearrange
from collections.abc import Callable, Iterable
from typing import Optional
import typing
import torch
import math
import numpy as np
import os
from ptflops import get_model_complexity_info


def _truncated_normal_init(weights: nn.Parameter):
    assert weights.ndim == 2
    std = (2 / (weights.shape[0] + weights.shape[1])) ** 0.5
    torch.nn.init.trunc_normal_(weights, std=std, a=-3 * std, b=3 * std)


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        _truncated_normal_init(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids_one_hot = torch.nn.functional.one_hot(
            token_ids, self.num_embeddings
        ).to(self.weight.dtype)
        return einsum(token_ids_one_hot, self.weight, "... v, v d -> ... d")


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = ((x**2).sum(-1, keepdim=True) / self.d_model + self.eps) ** 0.5
        result = x / norm * self.weight.to(x.dtype)
        return result.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_ff = d_ff or round(d_model * 8 / 3 / 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        _truncated_normal_init(self.w1.weight)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, dtype=dtype, device=device)
        _truncated_normal_init(self.w2.weight)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        _truncated_normal_init(self.w3.weight)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0
        self.max_seq_len = max_seq_len
        thetas = einsum(
            torch.arange(max_seq_len, dtype=torch.float32),
            torch.pow(theta, -2 * torch.arange(d_k // 2) / d_k),
            "i, k -> i k",
        ).to(device=device)
        self.register_buffer("sin", torch.sin(thetas), persistent=False)
        self.register_buffer("cos", torch.cos(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pos = torch.nn.functional.one_hot(token_positions, self.max_seq_len).to(x.dtype)
        sin = einsum(self.sin.to(x.dtype), pos, "i k, ... s i -> ... s k")
        cos = einsum(self.cos.to(x.dtype), pos, "i k, ... s i -> ... s k")
        x = rearrange(x, "... (k r) -> ... k r", r=2)
        x0, x1 = x[..., 0], x[..., 1]
        y = torch.stack(
            [
                x0 * cos - x1 * sin,
                x0 * sin + x1 * cos,
            ],
            dim=-1,
        )
        return rearrange(y, "... k r -> ... (k r)")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    exp = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scale = (query.shape[-1]) ** 0.5
    logits = einsum(query, key, "... s_q d_k, ... s_k d_k -> ... s_q s_k") / scale
    logits.masked_fill_(~mask, -torch.inf)
    scores = softmax(logits, dim=-1)
    return einsum(scores, value, "... s_q s_k, ... s_k d_v -> ... s_q d_v")


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(
            d_model, d_model, bias=False, dtype=dtype, device=device
        )
        _truncated_normal_init(self.q_proj.weight)
        self.k_proj = torch.nn.Linear(
            d_model, d_model, bias=False, dtype=dtype, device=device
        )
        _truncated_normal_init(self.k_proj.weight)
        self.v_proj = torch.nn.Linear(
            d_model, d_model, bias=False, dtype=dtype, device=device
        )
        _truncated_normal_init(self.v_proj.weight)
        self.output_proj = torch.nn.Linear(
            d_model, d_model, bias=False, dtype=dtype, device=device
        )
        _truncated_normal_init(self.output_proj.weight)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        else:
            print(token_positions)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "... s (h d_v) -> ... h s d_v", h=self.num_heads)
        k = rearrange(k, "... s (h d_v) -> ... h s d_v", h=self.num_heads)
        v = rearrange(v, "... s (h d_v) -> ... h s d_v", h=self.num_heads)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = token_positions.unsqueeze(-2) <= token_positions.unsqueeze(-1)
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... h s d_v -> ... s (h d_v)")
        return self.output_proj(attn)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        return z


class TransformerLM(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        vocab_size,
        context_length: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope=rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )
        _truncated_normal_init(self.lm_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def transformer_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
):
    print(f"vocab_size {vocab_size}")
    print(f"context_length {context_length}")
    print(f"num_layers {num_layers}")
    print(f"d_model {d_model}")
    print(f"num_heads {num_heads}")
    print(f"d_ff {d_ff}")
    num_embedding_params = vocab_size * d_model
    num_attn_params = d_model * d_model * 4 + d_model
    num_ffn_params = d_model * d_ff * 3 + d_model
    num_ln_final_params = d_model
    num_lm_head_params = d_model * vocab_size
    num_total_params = (
        num_embedding_params
        + (num_attn_params + num_ffn_params) * num_layers
        + num_ln_final_params
        + num_lm_head_params
    )
    print(f"total num params: {num_total_params}")
    print(f"total memory: {num_total_params * 2 / 2**30:.4f} GiB")

    embedding_flops = context_length * vocab_size * d_model * 2
    print(f"embedding_flops {embedding_flops / 1e12} tFLOPs")
    attn_proj_flops = context_length * d_model * d_model * 2 * 4 * num_layers
    print(f"attn_proj_flops {attn_proj_flops / 1e12} tFLOPs")
    attn_qk_flops = context_length * d_model * context_length * 2 * num_layers
    print(f"attn_kv_flops {attn_qk_flops / 1e12} tFLOPs")
    attn_v_flops = context_length * context_length * d_model * 2 * num_layers
    print(f"attn_v_flops {attn_v_flops / 1e12} tFLOPs")
    ffn_proj_flops = context_length * d_model * d_ff * 2 * 3 * num_layers
    print(f"ffn_proj_flops {ffn_proj_flops / 1e12} tFLOPs")
    output_flops = context_length * d_model * vocab_size * 2
    print(f"output_flops {output_flops / 1e12} tFLOPs")
    total_flops = (
        embedding_flops
        + attn_proj_flops
        + attn_qk_flops
        + attn_v_flops
        + ffn_proj_flops
        + output_flops
    )
    print(f"total_flops {total_flops / 1e12} tFLOPs")


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs -= inputs.max(dim=-1, keepdim=True)[0]
    log_z = torch.log(torch.exp(inputs).sum(dim=-1))
    targets = torch.nn.functional.one_hot(targets, inputs.shape[-1]).to(inputs.dtype)
    loss = -einsum(inputs, targets, "... v, ... v -> ...") + log_z
    return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: None = None):
        del closure
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                g = p.grad.data
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)

                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                alpha = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= alpha * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1


def cosine_learning_rate_schedule(
    t: int, lr_max: float, lr_min: float, t_warmup: int, t_cap: int
) -> float:
    if t < t_warmup:
        return t / t_warmup * lr_max
    if t <= t_cap:
        return lr_min + 0.5 * (
            1 + math.cos((t - t_warmup) / (t_cap - t_warmup) * math.pi)
        ) * (lr_max - lr_min)
    return lr_min


def gradient_clipping(params, l2_max, eps=1e-6):
    grad_sq = 0
    for p in params:
        if p.grad is None:
            continue
        grad_sq += (p.grad**2).sum()
    grad_l2 = grad_sq**0.5
    if grad_l2 >= l2_max:
        factor = l2_max / (grad_l2 + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= factor


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = []
    targets = []
    for start in starts:
        inputs.append(x[start : start + context_length])
        targets.append(x[start + 1 : start + context_length + 1])
    inputs = torch.tensor(inputs).to(device)
    targets = torch.tensor(targets).to(device)
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


if __name__ == "__main__":
    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr=1e3)
    # for t in range(10):
    #     opt.zero_grad()  # Reset the gradients for all learnable parameters.
    #     loss = (weights**2).mean()  # Compute a scalar loss value.
    #     print(loss.cpu().item())
    #     loss.backward()  # Run backward pass, which computes gradients.
    #     opt.step()  # Run optimizer step

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    context_length = 2048
    m = TransformerLM(
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        rope_theta=2000,
        vocab_size=50257,
        context_length=context_length,
        num_layers=48,
        device="cuda",
        dtype=torch.bfloat16,
    )
    num_total_params = sum(np.prod(p.shape) for p in m.parameters())
    print(f"total num params {num_total_params}")
    get_model_complexity_info(
        m,
        (1, context_length),
        input_constructor=lambda res: torch.ones(res, dtype=torch.int64).to("cuda"),
        as_strings=True,
        backend="pytorch",
        print_per_layer_stat=True,
        verbose=True,
    )

    transformer_accounting(
        vocab_size=50257,
        context_length=context_length,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
    )
