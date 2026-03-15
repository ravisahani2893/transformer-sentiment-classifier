import torch
import torch.nn as nn


# =============================================================================
# Baseline 1: Bag-of-Words Classifier
# =============================================================================
# Simplest possible text classifier.
# Completely ignores word order — "I love this movie" and
# "movie this love I" look identical to this model.
#
# Pipeline:
#   tokens → embeddings → average all → linear layers → prediction
# =============================================================================

class BoWClassifier(nn.Module):

    def __init__(self, vocab_size, embed_size, num_classes):
        super().__init__()

        # Same lookup table as transformer — word ID → vector
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Simple 2-layer classifier
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),   # 64 → 128
            nn.ReLU(),
            nn.Linear(embed_size * 2, num_classes),  # 128 → 2
        )

    def forward(self, x):
        # x shape: (N, seq_len)

        embedded = self.embedding(x)       # (N, seq_len, 64)

        # Average ALL token vectors into one sentence vector
        # This is the "bag" — order is thrown away completely
        pooled = embedded.mean(dim=1)      # (N, 64)

        return self.fc(pooled)             # (N, 2)


# =============================================================================
# Baseline 2: CNN Text Classifier
# =============================================================================
# Smarter than BoW — captures LOCAL word patterns (n-grams).
# A filter of size 2 sees pairs: "love this", "this movie"
# A filter of size 3 sees triplets: "I love this", "love this movie"
# But still has no global context like the Transformer.
#
# Pipeline:
#   tokens → embeddings → conv filters → max pooling → linear → prediction
# =============================================================================

class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embed_size, num_classes,
                 num_filters=64, kernel_sizes=[2, 3, 4]):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # One conv layer per kernel size
        # kernel_size=2 → looks at 2 words at a time (bigrams)
        # kernel_size=3 → looks at 3 words at a time (trigrams)
        # kernel_size=4 → looks at 4 words at a time (4-grams)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_size,     # each "channel" is one embedding dim
                out_channels=num_filters,   # how many patterns to detect
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        # After max pooling: num_filters per kernel_size, all concatenated
        # 64 filters × 3 kernel sizes = 192 features
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x shape: (N, seq_len)

        embedded = self.embedding(x)               # (N, seq_len, 64)

        # Conv1d expects (N, channels, length) — swap last two dims
        embedded = embedded.permute(0, 2, 1)       # (N, 64, seq_len)

        pooled_outputs = []

        for conv in self.convs:
            # Apply conv filter — slides across the sequence
            c = torch.relu(conv(embedded))         # (N, num_filters, seq_len - k + 1)

            # Max pooling — pick the strongest activation across the sequence
            # "Did this pattern appear ANYWHERE in the sentence?"
            p = c.max(dim=2).values                # (N, num_filters)

            pooled_outputs.append(p)

        # Concatenate all filter outputs
        out = torch.cat(pooled_outputs, dim=1)     # (N, num_filters * 3) = (N, 192)

        return self.fc(out)                        # (N, 2)
