import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

# hyperparams
batch_size = 64
block_size = 256
n_embed = 384
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2
# ---
torch.manual_seed(1337)
# ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
# ---
# Create a mapping (encoder and decoder)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])
data = torch.tensor(encode(text), dtype=torch.long)
n= int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# ---

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.atten(x).split(n_embed, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v  # B nh T T @ B nh T hs -> B nh T hs
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        out = self.resid_dropout(self.c_proj(out))
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embed, 4 * n_embed),
            c_proj=nn.Linear(4 * n_embed, n_embed),
            ac=nn.ReLU(),
            drop=nn.Dropout(dropout),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.drop(m.c_proj(m.ac(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_1(x))
        return x


class small_GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embed),  # token
            wpe=nn.Embedding(block_size, n_embed),  # position
            drop=nn.Dropout(0.05),
            h=nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embed),
        ))
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)  # B,T,C #We want to arrange this into b,t,c=embded
        pos_emb = self.transformer.wpe(torch.arange(T, device=device))  # ints from 0 to T-1 #T,C
        x = self.transformer.drop(tok_emb + pos_emb)  # B T
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # b,t,c c=vocab size
        # Batch=4, Time=8(block), Channels=vocab_size=65
        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions : )))
            idx_cond = idx[:, -block_size:]  # crop context
            logits, loss = self(idx_cond)
            # FOCUS ONLY ON THE LASST TIME STEP

            logits = logits[:, -1, :]  # becomes B,C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # B,C
            idx_next = torch.multinomial(probs, num_samples=1)  # B,1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx

def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)

def load_model(PATH):
    model = small_GPT()
    model = model.to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200
# TRAINING
m=small_GPT()
m = m.to(device)
optim = torch.optim.AdamW(m.parameters(), lr=lr)
model_path = os.path.join( "./SaveModel", "model"+".pth")
def get_batch(split):
    #Generate a small batch of data of inputs x and targets y
    data =  train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #ix is 4 numbers that will be randmoly generated between 0 and len(data)-blocksize
    #Random off sets
    x = torch.stack([data[i:i+block_size] for i in ix]) #stack them as rows
    y = torch.stack([data[i+1: i+block_size+1] for i in ix]) #stack them as rows
    #Will become a 4 x 8 tensor
    return x,y

for steps in range(1000):
    xb,yb = get_batch('train')
    xb = xb.to(device)
    yb = yb.to(device)
    logits, loss = m(xb,yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    print(loss.item())
torch.save(m.state_dict(), model_path)

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()) )


