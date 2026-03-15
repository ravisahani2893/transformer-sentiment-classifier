import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SentimentDataset
from model import TransformerClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = [
("I love this movie",1),
("This film is amazing",1),
("I enjoyed this film",1),
("Amazing movie",1),
("Great film",1),

("This movie is terrible",0),
("I hate this movie",0),
("This film is boring",0),
("Awful movie",0),
("Bad film",0),
]



sentences = [d[0] for d in data]

words = set(" ".join(sentences).lower().split())

word2idx = {w: i+1 for i, w in enumerate(words)}

vocab_size = len(word2idx) + 1

vocab_size
word2idx
data
dataset = SentimentDataset(data, word2idx, seq_len=10)

loader = DataLoader(dataset, batch_size=2, shuffle=True)


model = TransformerClassifier(
    vocab_size=vocab_size,
    embed_size=64,
    num_layers=2,
    heads=4,
    num_classes=2
)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 20

def visualize_attention(tokens, attention):

    attn = attention[0,0].detach().numpy()   # first sentence, first head

    plt.figure(figsize=(6,5))

    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True
    )

    plt.title("Attention Heatmap (Head 0)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    plt.show()


def predict(sentence, model, word2idx, seq_len=10):

    model.eval()

    tokens = sentence.lower().split()
    

    ids = [word2idx.get(w, 0) for w in tokens]

    if len(ids) < seq_len:
        ids += [0] * (seq_len - len(ids))

    ids = ids[:seq_len]

    x = torch.tensor(ids).unsqueeze(0)

    with torch.no_grad():
        output, attention = model(x)
    print("\nTokens:", tokens)

    print("\nAttention Matrix (Head 0):")
    print(attention[0,0])
    visualize_attention(tokens, attention)
    pred = torch.argmax(output, dim=1).item()

    if pred == 1:
        print("Positive 🙂")
    else:
        print("Negative ☹️")

for epoch in range(epochs):

    total_loss = 0

    for x, y in loader:

        outputs, attention = model(x)

        loss = criterion(outputs, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


predict("I really love this film", model, word2idx)
predict("This movie is horrible", model, word2idx)
predict("The film was boring", model, word2idx)
predict("Amazing movie", model, word2idx)