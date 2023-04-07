#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.nn import functional as F
import torch.nn as nn
import torch
import os
import re
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
block_size = 256
dropout = 0.2
maxiters = 10000
eval_interval = 500
eval_iters = 200
# set n_embed to 256 for faster training
n_embed = 384
n_head = 6
n_layer = 6
learning_rate = 3e-4
print(device)
#chat = open("../_chat.txt", encoding="utf-8")
with open("C:/Users/dogut/OneDrive/Masaüstü/HeaderFolder/_chat.txt",  encoding="utf-8") as f:
    chat = [line.strip('\n') for line in f]

joinedChat = ' '.join(chat)
print(joinedChat[:500])

emoj = re.compile("["
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  u"\U00002500-\U00002BEF"  # chinese char
                  u"\U00002702-\U000027B0"
                  u"\U00002702-\U000027B0"
                  u"\U000024C2-\U0001F251"
                  u"\U0001f926-\U0001f937"
                  u"\U00010000-\U0010ffff"
                  u"\u2640-\u2642"
                  u"\u2600-\u2B55"
                  u"\u200d"
                  u"\u23cf"
                  u"\u23e9"
                  u"\u231a"
                  u"\ufe0f"  # dingbats
                  u"\u3030"
                  "]+", re.UNICODE)
joinedChat = (emoj.sub(r'', joinedChat))  # no emoji

joinedChat = re.sub(r"\[[^\]]*\]", '\n', joinedChat)
joinedChat = re.sub(r" Elif Yavrum:", 'Elif Yavrum:', joinedChat)
joinedChat = re.sub(r" Dogu Tamgac:", 'Dogu Tamgac:', joinedChat)
joinedChat = joinedChat.replace("\u200e", "")
print(joinedChat[:500])

chars = sorted(list(set(joinedChat)))

vocab_size = len(set(chars))
print(''.join(chars))
print(vocab_size)


# # Building character level language model. <br>
# stoi: Map strings to integers. <br>
# Then apply it and create encoding and decoding lambda functions <br>
# (Normally you can use sentencepiece from google or tiktoken)


stoi = {ch: i for i, ch in enumerate(chars)}
itoi = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[i] for i in s]
def decode(l): return ''.join([itoi[i] for i in l])


print(encode("l l"))
print(decode(encode("l l")))

torch.cuda.is_available()
data = torch.tensor(encode(joinedChat), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

n = int(0.95 * len(data))
train_data = data[:n]
val_data = data[n:]


# Now take the data asa blocks. We chose the block size 8 and initialized the length of the training data with 9 as there are 8 positions to guess


train_data[:block_size+1]


torch.manual_seed(1337)  # to be able to reproduce


def get_batch(split):
    data = train_data if split == 'train' else val_data
    # shape batchsize row vector
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stacked multiple sentences as row vectors
    x = torch.stack([data[i:i+block_size] for i in ix])
    # to train
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch('train')
xb = xb.to(device)
yb = yb.to(device)
print("input:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

# no back propagation


@torch.no_grad()
def estimate_loss():
    out = {}
    # normally no need but good practive, freezes batchnorm and dropout layers
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# # Using Bigram model=> Simple fast easy to understand
# Predict the next word from the earlier word.(In our case character) Conditional probability


torch.manual_seed(1337)

vocab_size = len(set(chars))
print(vocab_size)
# get the earlier character then propagate it


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # print(k)
        # only transpose t and c not the B
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # B T T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # over the batches one by one
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei@v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,  n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # wrapper essentialy a table by number of characters*number of characters where you have the scores
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        #print("tokentable", self.token_embedding_table)
        #self.sa_heads = MultiHeadAttention(4, (n_embed)//4)
        #self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (t,c)
        x = tok_embed+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        #x = self.sa_heads(x)
        #x = self.ffwd(x)
        logits = self.lm_head(x)  # BTvocabsize
        # logits B t C where B=4 batchsize, T=8 chunks, C=number of characters
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # remove Chunk indexes like appending together
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss  # scores

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # Only take the last characters from chunks
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # get one sample from the softmax
            idx_next = torch.multinomial(
                probs, num_samples=1)  # (B,1)
            # append the word to the chunk
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


model = BigramLanguageModel()
model = model.to(device)
logits, loss = model(xb, yb)  # forward
print(logits.shape)
# estimated loss ln(1/131)

#print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long),max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(maxiters):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(losses)
        print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long).to(
            device), max_new_tokens=400)[0].tolist()))
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long).to(
    device), max_new_tokens=10000)[0].tolist()))
print("finalloss:", loss.item())
