import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import SentimentDataset
from model import TransformerClassifier
from baselines import BoWClassifier, CNNClassifier


# =============================================================================
# Data
# =============================================================================

data = [
    ("I love this movie",       1),
    ("This film is amazing",    1),
    ("I enjoyed this film",     1),
    ("Amazing movie",           1),
    ("Great film",              1),
    ("This movie is terrible",  0),
    ("I hate this movie",       0),
    ("This film is boring",     0),
    ("Awful movie",             0),
    ("Bad film",                0),
]

sentences  = [d[0] for d in data]
words      = set(" ".join(sentences).lower().split())
word2idx   = {w: i+1 for i, w in enumerate(words)}
vocab_size = len(word2idx) + 1

dataset = SentimentDataset(data, word2idx, seq_len=10)
loader  = DataLoader(dataset, batch_size=2, shuffle=True)


# =============================================================================
# Models
# =============================================================================

EMBED_SIZE  = 64
NUM_CLASSES = 2

models = {
    "Bag-of-Words": BoWClassifier(
        vocab_size  = vocab_size,
        embed_size  = EMBED_SIZE,
        num_classes = NUM_CLASSES,
    ),
    "CNN": CNNClassifier(
        vocab_size   = vocab_size,
        embed_size   = EMBED_SIZE,
        num_classes  = NUM_CLASSES,
        num_filters  = 64,
        kernel_sizes = [2, 3, 4],
    ),
    "Transformer": TransformerClassifier(
        vocab_size  = vocab_size,
        embed_size  = EMBED_SIZE,
        num_layers  = 2,
        heads       = 4,
        num_classes = NUM_CLASSES,
    ),
}


# =============================================================================
# Training
# =============================================================================

EPOCHS    = 20
criterion = nn.CrossEntropyLoss()
results   = {}

for name, model in models.items():

    print(f"\n{'='*45}")
    print(f"  Training: {name}")
    print(f"{'='*45}")

    optimizer    = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []

    for epoch in range(EPOCHS):

        total_loss = 0

        for x, y in loader:

            # Transformer returns (output, attention), others return just output
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_history.append(total_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {total_loss:.4f}")

    results[name] = {
        "model":        model,
        "loss_history": loss_history,
    }


# =============================================================================
# Accuracy on Training Data
# =============================================================================

print(f"\n{'='*45}")
print(f"  Accuracy on Training Data")
print(f"{'='*45}")

for name, r in results.items():

    model = r["model"]
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            preds    = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    pct = 100 * correct / total
    bar = "█" * int(pct // 5)    # simple ASCII bar

    print(f"  {name:15s} → {correct:2d}/{total}  ({pct:.0f}%)  {bar}")


# =============================================================================
# Predictions on New Sentences
# =============================================================================

test_sentences = [
    "I really love this film",
    "This movie is horrible",
    "The film was boring",
    "Amazing movie",
    "I enjoyed watching this",
    "Terrible and awful film",
]

label_map = {0: "Negative ☹️ ", 1: "Positive 🙂"}


def encode(sentence, word2idx, seq_len=10):
    tokens = sentence.lower().split()
    ids    = [word2idx.get(w, 0) for w in tokens]
    if len(ids) < seq_len:
        ids += [0] * (seq_len - len(ids))
    return torch.tensor(ids[:seq_len]).unsqueeze(0)  # (1, seq_len)


print(f"\n{'='*45}")
print(f"  Predictions on Unseen Sentences")
print(f"{'='*45}")

for sentence in test_sentences:
    x = encode(sentence, word2idx)
    print(f"\n  '{sentence}'")
    for name, r in results.items():
        model = r["model"]
        model.eval()
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
        pred = torch.argmax(output, dim=1).item()
        print(f"    {name:15s} →  {label_map[pred]}")


# =============================================================================
# Loss Curve Comparison Plot
# =============================================================================

plt.figure(figsize=(9, 5))

colors = {
    "Bag-of-Words": "#e74c3c",
    "CNN":          "#f39c12",
    "Transformer":  "#2ecc71",
}

for name, r in results.items():
    plt.plot(
        range(1, EPOCHS + 1),
        r["loss_history"],
        label=name,
        color=colors[name],
        linewidth=2,
        marker="o",
        markersize=3,
    )

plt.title("Training Loss Comparison",  fontsize=14)
plt.xlabel("Epoch",                    fontsize=12)
plt.ylabel("Total Loss",               fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150)
plt.show()
print("\n  Loss curve saved → loss_comparison.png")
