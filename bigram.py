# Bigram model
import torch
import torch.nn as nn
from torch.nn import functional as F
# ---
torch.manual_seed(1337)
# ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
# ---
# hyperparams
batch_size = 64
block_size = 256
n_embed = 384
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.2
# ---
# Encoding, decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])
# ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (
    batch_size,))  # ix is 4 numbers that will be randmoly generated between 0 and len(data)-blocksize
    # Random off sets
    x = torch.stack([data[i:i + block_size] for i in ix])  # stack them as rows
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])  # stack them as rows
    x, y = x.to(device), y.to(device)
    # Will become a 4 x 8 tensor
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # BTC @ BCT -> BTT
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # BTT
        wei = F.softmax(wei, dim=-1)  # BTT
        wei = self.dropout(wei)
        out = wei @ v  # BTT @ BTC -> BTC
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # over channels
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), #4 just comes from paper
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),#Acts as our proj from multihead
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ##TRANSFORMER BLOCK
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)  # num_embedding, embdedding_dim
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # chooses a row corresponding to index
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C #We want to arrange this into b,t,c=embded
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # ints from 0 to T-1 #T,C
        x = tok_emb + pos_emb  # B T
        x = self.blocks(x)
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


model = BigramLanguageModel()
m = model.to(device)
