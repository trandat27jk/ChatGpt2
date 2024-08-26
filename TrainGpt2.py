import inspect
import math
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

#Parallel traininging
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
import os

def ddp_setup(rank,world_size):
    os.environ['MASTER_HOST']="localhost"
    os.environ['MASTER_PORT']='12355'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)
    
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {B*T} tokens per epoch")
        print(f"Epoch 1: {len(self.tokens//(B*T))} per batchs")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T) > len(self.tokens):
            self.current_position = 0

        return x, y


# config
@dataclass
class DATGPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.ln2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to_qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.to_out = nn.Linear(config.n_embd, config.n_embd)
        self.to_out.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.head, C // self.head).transpose(1, 2)
        q = q.view(B, T, self.head, C // self.head).transpose(1, 2)
        v = v.view(B, T, self.head, C // self.head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.to_out(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.mlp(x))
        return x


class DATGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=nn.LayerNorm(config.n_embd),
            )
        )
        self.to_logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.layers.wte.weight = self.to_logits.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number params decay: {num_decay_params}")
        print(f"number nodecay_params: {num_nodecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def forward(self, x, target=None):
        B, T = x.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos = self.layers.wpe(pos)
        tokens_enc = self.layers.wte(x)
        x = tokens_enc + pos
        for block in self.layers.h:
            x = block(x)
        x = self.layers.norm(x)
        logits = self.to_logits(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
total_batch_size = 524288
B = 4
T = 32
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)


# warm up learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 100


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


train_loader = DataLoaderLite(B=B, T=T)
model = DATGPT(DATGPTConfig()).to(device)
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device
)
torch.set_float32_matmul_precision("high")
for i in range(max_steps):
    optimizer.zero_grad()
    t0 = time.time()
    loss_acc = 0.0
    for i in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_acc += loss.detach()
        loss.backward()
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000
    token_per_sec = (train_loader.B * train_loader.T) / duration
    print(f"loss:{loss_acc},dt:{duration} ms, tok/ken{token_per_sec}")
