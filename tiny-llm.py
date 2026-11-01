import random
import torch
import torch.nn.functional as F
from torch import nn

# ---- your tiny training text ----
text = """
The quick brown fox jumps over the lazy dog.
A tiny model learns to predict the next token.
Language models are just probability machines.
This is a demo of a very small LLM.
"""

# ---- build vocab ----
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

# ---- training setup ----
block_size = 8
hidden_size = 32
embedding_size = 16

def make_examples(text, block_size):
    xs, ys = [], []
    for i in range(len(text) - block_size):
        x = [stoi[ch] for ch in text[i:i + block_size]]
        y = stoi[text[i + block_size]]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

X, Y = make_examples(text, block_size)

class TinyLM(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_size, hidden_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.net = nn.Sequential(
            nn.Linear(block_size * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        # x: (batch, block_size)
        emb = self.embedding(x)  # (batch, block_size, embedding_size)
        flat = emb.reshape(x.shape[0], -1)  # flatten context
        return self.net(flat)  # (batch, vocab_size)

model = TinyLM(vocab_size, block_size, embedding_size, hidden_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---- training ----
batch_size = 32
for step in range(3000):
    idx = torch.randint(0, X.shape[0], (batch_size,))
    xb = X[idx]
    yb = Y[idx]

    logits = model(xb)
    loss = F.cross_entropy(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step:4d} loss = {loss.item():.4f}")

# ---- generation ----
def generate(prompt, length=80):
    context = prompt[-block_size:]
    out = prompt

    for _ in range(length):
        x = torch.tensor([stoi[ch] for ch in context[-block_size:]], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits[0], dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
        next_char = itos[next_idx]
        out += next_char
        context = out[-block_size:]
    return out

print("\nGenerated text:")
print(generate("The", 100))