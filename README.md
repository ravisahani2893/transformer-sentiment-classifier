# Transformer Sentiment Classifier

A from-scratch implementation of a Transformer encoder for binary sentiment classification (Positive / Negative) built with PyTorch. No pretrained models — every component is written and explained explicitly.

---

## Project Structure

```
transformer-sentiment-classifier/
├── dataset.py   # Tokenization, encoding, and PyTorch Dataset
├── model.py     # Full Transformer architecture (Attention, Encoder, Classifier)
├── train.py     # Training loop, optimizer, loss, attention visualization
└── predict.py   # (placeholder for standalone inference)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch matplotlib seaborn
```

### 2. Train and predict

```bash
python train.py
```

This will:
- Train for 20 epochs on 10 toy movie review sentences
- Print loss per epoch
- Run predictions on 4 test sentences
- Display an attention heatmap for each prediction

---

## Architecture Overview

```
Raw Text
   ↓
Tokenize  →  "I love this movie"  →  [3, 7, 1, 4, 0, 0, 0, 0, 0, 0]
   ↓
Embedding  →  each token ID → 64-dim learned vector      shape: (batch, seq_len, 64)
   ↓
Positional Encoding  →  add position signal              shape: (batch, seq_len, 64)
   ↓
TransformerBlock × 2
  ├── Multi-Head Self-Attention                          shape: (batch, seq_len, 64)
  ├── Add & Norm  (residual connection + LayerNorm)
  ├── Feed Forward Network  (64 → 256 → 64)
  └── Add & Norm
   ↓
Mean Pooling  →  average all token vectors               shape: (batch, 64)
   ↓
Linear Classifier  →  64 → 2 logits                     shape: (batch, 2)
   ↓
Prediction: Positive / Negative
```

---

## Code Walkthrough

### `dataset.py` — Preparing Data

#### The Problem
Neural networks only understand numbers, not words. Every word must be converted to an integer ID before it can enter the model.

```python
word2idx = {"love": 1, "movie": 2, "terrible": 3, ...}
"I love this movie" → [5, 1, 8, 2]
```

#### `SentimentDataset`

Inherits from PyTorch's `Dataset` base class. This contract requires two methods:

| Method | Purpose |
|--------|---------|
| `__len__` | Returns total number of samples |
| `__getitem__(idx)` | Returns one sample `(x, y)` at index `idx` |

PyTorch's `DataLoader` uses these to automatically batch, shuffle, and feed data during training.

#### `encode(sentence)`

```python
def encode(self, sentence):
    tokens = sentence.lower().split()                    # "I Love" → ["i", "love"]
    ids = [self.word2idx.get(w, 0) for w in tokens]     # ["i", "love"] → [5, 1]

    # Pad with zeros if sentence is shorter than seq_len
    if len(ids) < self.seq_len:
        ids += [0] * (self.seq_len - len(ids))           # [5, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    return ids[:self.seq_len]                            # truncate if longer
```

**Why fixed length?**
Batches are processed as matrices. A matrix must be rectangular — all rows the same length. Padding with `0` fills shorter sentences. Longer sentences are truncated.

**`word2idx.get(w, 0)`** — if a word is not in the vocabulary, it defaults to `0` (unknown token) instead of crashing.

---

### `model.py` — The Transformer

#### `PositionalEncoding` — *"Where am I in the sentence?"*

**The Problem:**
Unlike RNNs that process tokens one by one, Transformers process ALL tokens simultaneously. This means the model has no idea about word order. "Dog bites man" and "Man bites dog" would look identical.

**The Solution:**
Add a unique positional signal to each token's vector using sine and cosine waves at different frequencies.

```python
pe[pos, i]   = sin(pos / 10000^(2i/embed_size))
pe[pos, i+1] = cos(pos / 10000^(2i/embed_size))
```

Properties of this encoding:
- Every position gets a **unique** vector
- Nearby positions have **similar** vectors (smooth transition)
- The model can infer **relative distance** between tokens
- Works for any sequence length

```python
def forward(self, x):
    return x + self.pe[:, :x.shape[1]]   # simply add position info to each word vector
```

---

#### `SelfAttention` — *The Core Mechanism*

**Core Question:** *"When understanding this word, which other words should I pay attention to?"*

Example: In *"The bank by the river was flooded"*, the word "bank" should attend strongly to "river". Self-attention learns these relationships automatically from data.

##### Query, Key, Value — The Library Analogy

| Component | Analogy | Role |
|-----------|---------|------|
| **Query (Q)** | Your search query | What this token is looking for |
| **Key (K)** | Book index / title | What each token advertises it contains |
| **Value (V)** | Book content | The actual information a token carries |

A token's query is matched against every other token's key. The score determines how much of each token's value to incorporate.

##### Step-by-Step Forward Pass

**Step 1: Linear projections**
```python
values  = self.values(x)    # (N, seq_len, 64)
keys    = self.keys(x)      # (N, seq_len, 64)
queries = self.queries(x)   # (N, seq_len, 64)
```
Three separate learned transformations give each token three different "roles".

**Step 2: Split into multiple heads**
```python
values = values.reshape(N, seq_len, heads, head_dim)
# (N, 10, 64) → (N, 10, 4, 16)  — 4 heads, each 16-dim
```
Each head learns to attend to different relationship types (grammar, semantics, sentiment, etc.) in parallel.

**Step 3: Permute for matrix multiplication**
```python
Q = queries.permute(0, 2, 1, 3)  # (N, 10, 4, 16) → (N, 4, 10, 16)
K = keys.permute(0, 2, 1, 3)
V = values.permute(0, 2, 1, 3)
```

**Step 4: Compute attention scores — QKᵀ**
```python
energy = torch.matmul(Q, K.transpose(-2, -1))
# (N, 4, 10, 16) × (N, 4, 16, 10) → (N, 4, 10, 10)
#                                              ↑↑↑↑
#                              every token scored against every other token
```

**Step 5: Scale**
```python
energy = energy / math.sqrt(self.head_dim)   # divide by √16 = 4
```
Prevents large dot products from making softmax near one-hot, which kills gradients.

**Step 6: Softmax → attention weights**
```python
attention = torch.softmax(energy, dim=-1)   # each row sums to 1.0
```

```
          "I"    "love"  "this"  "movie"
"I"      [0.05,  0.70,   0.10,   0.15]   ← "I" attends mostly to "love"
"love"   [0.10,  0.20,   0.10,   0.60]   ← "love" attends mostly to "movie"
```

**Step 7: Weighted sum of Values**
```python
out = torch.matmul(attention, V)
# (N, 4, 10, 10) × (N, 4, 10, 16) → (N, 4, 10, 16)
```

**Step 8: Merge heads and project**
```python
out = out.permute(0, 2, 1, 3).reshape(N, seq_len, embed_size)
# (N, 4, 10, 16) → (N, 10, 64)
return self.fc_out(out), attention
```

---

#### `TransformerBlock` — *One Full Encoder Layer*

```
Input x
  │
  ├──→ SelfAttention(x) ──→ attention_out
  │                               │
  └────────── + ←────────────────┘   ← Residual connection
              │
           LayerNorm  →  x
              │
           FeedForward(x)  →  fwd
              │
  ┌────── + ←─┘   ← Residual connection
  │
LayerNorm  →  output
```

**Residual Connections (`+ x`)**
The original input is added back to each sublayer's output. Each layer only learns the *correction* needed, not a full transformation. Critical for:
- Preventing gradient vanishing in deep networks
- Faster, more stable training

**Feed Forward Network**
```python
nn.Linear(64, 256) → ReLU → nn.Linear(256, 64)
```
- Attention mixes information **across tokens**
- FFN processes each token **independently** with more capacity
- The 4× expansion (64 → 256) gives a "working space" for complex transformations
- ReLU introduces non-linearity — without it, stacking linear layers is still just one linear layer

**LayerNorm**
Normalizes each token vector to mean≈0, std≈1 after every sublayer. Stabilizes training.

---

#### `TransformerClassifier` — *The Full Model*

```python
self.embedding  = nn.Embedding(vocab_size, embed_size)   # learned lookup table
self.position   = PositionalEncoding(embed_size)          # positional signal
self.layers     = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
self.fc_out     = nn.Linear(embed_size, num_classes)      # final classifier head
```

**`nn.Embedding`** — a lookup table of shape `[vocab_size, 64]`. Token ID `5` → row 5 = a 64-dim vector. These vectors are *learned* during training. Similar words converge to similar vectors.

**`nn.ModuleList`** — a Python list that PyTorch tracks. Required instead of a plain list so the optimizer knows about all nested parameters.

**Forward Pass:**
```python
def forward(self, x):
    out = self.embedding(x)    # (batch, 10) → (batch, 10, 64)
    out = self.position(out)   # add positional encoding
    for layer in self.layers:  # pass through N transformer blocks
        out, attn = layer(out)
    out = out.mean(dim=1)      # (batch, 10, 64) → (batch, 64)  ← mean pooling
    return self.fc_out(out), attn  # → (batch, 2)
```

**Mean Pooling** — averages all 10 token vectors into one 64-dim sentence representation. The final linear layer maps this to 2 logits (scores for negative and positive).

---

### `train.py` — Training the Model

#### Setup

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**CrossEntropyLoss** — standard loss for classification. Softmaxes the 2 logits → `-log(probability of correct class)`. Confident wrong predictions are penalized heavily.

**Adam** — adaptive optimizer. Adjusts learning rate per parameter using momentum. Near-universal default for transformer training.

#### The Training Loop

```python
for epoch in range(epochs):
    for x, y in loader:

        outputs, attention = model(x)    # 1. Forward pass
        loss = criterion(outputs, y)     # 2. Compute loss

        optimizer.zero_grad()            # 3a. Clear old gradients
        loss.backward()                  # 3b. Backpropagation
        optimizer.step()                 # 4.  Update all parameters
```

| Step | What happens |
|------|-------------|
| `model(x)` | Input flows forward through the full network |
| `criterion(outputs, y)` | Measures how wrong the predictions are |
| `zero_grad()` | Clears gradients from previous batch (PyTorch accumulates by default) |
| `loss.backward()` | Computes ∂loss/∂every_parameter via chain rule |
| `optimizer.step()` | Nudges every parameter to reduce loss |

#### Attention Visualization

```python
def visualize_attention(tokens, attention):
    attn = attention[0, 0].detach().numpy()   # sentence 0, head 0 → 10×10 array
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens)
```

After training, inspect which tokens attend to which. A trained model shows "love" strongly attending to "movie" in positive reviews — learned purely from data.

---

## Complete Data Flow — Shape Tracking

```
Input:                    (2, 10)          ← 2 sentences, 10 token IDs each
After Embedding:          (2, 10, 64)      ← each token → 64-dim vector
After Position:           (2, 10, 64)      ← positional info added (same shape)

─── Inside SelfAttention ────────────────────────────────────────────
After Linear Q/K/V:       (2, 10, 64)
After reshape:            (2, 10,  4, 16)  ← split 64 dims into 4 heads × 16
After permute:            (2,  4, 10, 16)  ← heads before seq_len
QKᵀ energy:               (2,  4, 10, 10)  ← every token vs every token
After softmax:            (2,  4, 10, 10)  ← attention weights, rows sum to 1
After × Values:           (2,  4, 10, 16)
After merge heads:        (2, 10, 64)      ← heads concatenated back
─────────────────────────────────────────────────────────────────────

After 2× TransformerBlock:    (2, 10, 64)
After mean pooling:           (2, 64)      ← one vector per sentence
After fc_out:                 (2,  2)      ← logits: [neg_score, pos_score]
```

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `embed_size` | 64 | Dimension of every token vector |
| `num_layers` | 2 | Number of stacked TransformerBlocks |
| `heads` | 4 | Number of attention heads per layer |
| `head_dim` | 16 | `embed_size / heads` — dimension per head |
| `forward_expansion` | 4 | FFN hidden size = `4 × embed_size` = 256 |
| `seq_len` | 10 | Fixed input length (pad/truncate to this) |
| `num_classes` | 2 | Negative (0) or Positive (1) |
| `lr` | 0.001 | Adam learning rate |
| `batch_size` | 2 | Sentences processed per gradient update |
| `epochs` | 20 | Full passes over training data |

---

## Key Concepts Reference

| Concept | One-Line Intuition |
|---------|-------------------|
| **Embedding** | Learned dictionary: integer ID → dense meaning vector |
| **Positional Encoding** | Sine/cosine fingerprint telling each token its position |
| **Query / Key / Value** | Search query / index / content — for soft information retrieval |
| **Scaled Dot-Product Attention** | Score every token pair, softmax, weighted-sum the values |
| **Multi-Head Attention** | Run attention in parallel across multiple learned subspaces |
| **Residual Connection** | Skip highway: add input back so layers learn corrections, not rewrites |
| **LayerNorm** | Normalize each vector to stabilize training |
| **Feed Forward Network** | Per-token transformation with extra capacity (4× expansion) |
| **Mean Pooling** | Average all token vectors into one sentence representation |
| **CrossEntropyLoss** | Penalizes confident wrong predictions more than uncertain ones |
| **Backpropagation** | Chain rule applied across the whole graph to compute all gradients |
| **Adam** | Adaptive optimizer — adjusts learning rate per parameter using momentum |

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Transformer paper (2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — best visual explanation
- [PyTorch Tutorials](https://pytorch.org/tutorials/) — official beginner guides
- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — builds a transformer from scratch on video
