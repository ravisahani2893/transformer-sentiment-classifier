import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):

        N, seq_len, embed_size = x.shape
        # print("\nInput shape:", x.shape)

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # print("After linear layers:")
        # print("values:", values.shape)
        # print("keys:", keys.shape)
        # print("queries:", queries.shape)

        # Split embedding into multiple heads
        values = values.reshape(N, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(N, seq_len, self.heads, self.head_dim)
        queries = queries.reshape(N, seq_len, self.heads, self.head_dim)

       

        # Reorder dimensions for attention
        Q = queries.permute(0, 2, 1, 3)
        K = keys.permute(0, 2, 1, 3)
        V = values.permute(0, 2, 1, 3)

        # print("\nAfter permute:")
        # print("Q:", Q.shape)
        # print("Q visualise:", Q)
        # print("K:", K.shape)
        # print("K visualise:", K)
        # print("V:", V.shape)
        # print("V visualise:", V)

        # Compute attention scores (QK^T)
        energy = torch.matmul(Q, K.transpose(-2, -1))

        # print("\nEnergy shape (QK^T):", energy.shape)

        # Scale
        energy = energy / math.sqrt(self.head_dim)

        # Softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        # print("\nAttention shape:", attention.shape)

        # Print attention matrix for first sentence and first head
        # print("\nAttention matrix (sentence 0, head 0):")
        # print(attention[0, 0])

        # Multiply attention weights with values
        out = torch.matmul(attention, V)

    

        # Restore original order
        out = out.permute(0, 2, 1, 3)



        # Combine heads
        out = out.reshape(N, seq_len, self.embed_size)

        # Final linear layer
        return self.fc_out(out), attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
    def forward(self, x):

        # Self Attention
        attention_out, attn_weights = self.attention(x)

        # Residual connection + normalization
        x = self.norm1(attention_out + x)

        # Feed Forward Network
        forward = self.feed_forward(x)

        # Residual connection + normalization
        out = self.norm2(forward + x)

        return out, attn_weights

class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_length=100):
        super().__init__()

        pe = torch.zeros(max_length, embed_size)

        for pos in range(max_length):
            for i in range(0, embed_size, 2):

                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/embed_size)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):

        seq_len = x.shape[1]

        return x + self.pe[:, :seq_len]

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        num_classes,
        max_length=100
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.position = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):

        # x shape: (batch_size, seq_len)

        out = self.embedding(x)

        out = self.position(out)

        attn = None

        for layer in self.layers:
            out, attn = layer(out)

        # mean pooling across sequence
        out = out.mean(dim=1)

        return self.fc_out(out), attn



    